#!/usr/bin/env python3
"""Cache frozen AudioSep-CLAP controller inputs without oracle labels.

The cache has two disjoint, inference-safe payloads:

* one full-question CLAP text embedding per manifest sample ID; and
* one effective 32-frame CLAP audio sequence per deduplicated scene ID.

Only waveform paths, questions, and non-semantic identifiers are read from the
manifest.  Event labels, answers, evidence annotations, and oracle prompts are
never copied into any output artifact.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from mixi_understanding.data.qces_real10_schema import (
    INFERENCE_SCHEMA_VERSION as REAL10_INFERENCE_SCHEMA_VERSION,
    canonical_inference_manifest_fingerprint,
    parse_inference_manifest,
    resolve_manifest_mixture_path,
)


QUESTION_CACHE_FORMAT = "qces_audiosep_clap_question_features_v1"
SCENE_CACHE_FORMAT = "qces_audiosep_clap_scene_features_v1"
RECEIPT_FORMAT = "qces_audiosep_clap_frozen_controller_cache_receipt_v1"
FEATURE_SPACE = "audiosep_checkpoint_query_encoder_clap_joint_512"

CANONICAL_SAMPLE_RATE = 32_000
CANONICAL_DURATION_SECONDS = 10.0
CANONICAL_NUM_SAMPLES = 320_000
CLAP_SAMPLE_RATE = 48_000
CLAP_NUM_SAMPLES = 480_000
RAW_FINE_FRAMES = 1_024
EFFECTIVE_FRAMES = 32
REPEAT_RATIO = RAW_FINE_FRAMES // EFFECTIVE_FRAMES
FINE_DIM = 1_024
JOINT_DIM = 512
EXPECTED_QUERY_STATE_TENSORS = 505
VALID_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})
SOURCE_SUFFIXES = (".json", ".py", ".yaml", ".yml")
NONPERSISTENT_POSITION_ID_SUFFIX = "embeddings.position_ids"
REAL10_CANONICAL_WAV_FORMAT = "WAV"
REAL10_CANONICAL_WAV_SUBTYPE = "FLOAT"


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    scene_id: str
    question: str


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    mixture_path: Path
    mixture_manifest_path: str
    sample_rate: int
    num_samples: int
    duration_seconds: float
    declared_mixture_sha256: str | None = None
    dataset_root: Path | None = None


@dataclass(frozen=True)
class CacheManifestSpec:
    """Parsed cache inputs plus an optional sealed QCES-Real-10 contract."""

    samples: tuple[SampleSpec, ...]
    scenes: tuple[SceneSpec, ...]
    schema_versions: tuple[str, ...]
    real10_inference_manifest_fingerprint: str | None = None

    @property
    def is_real10_inference(self) -> bool:
        return self.real10_inference_manifest_fingerprint is not None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="CUDA is the paper-cache default; CPU remains available for audits.",
    )
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def source_tree_identity(root: Path) -> dict[str, Any]:
    """Hash every AudioSep source/config file that can define CLAP behavior."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    paths = sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(resolved).parts
        and ".cache" not in path.relative_to(resolved).parts
        and "__pycache__" not in path.relative_to(resolved).parts
        and path.suffix.lower() in SOURCE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"no AudioSep source/config files found under {resolved}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "hashed_file_count": len(paths),
        "included_suffixes": list(SOURCE_SUFFIXES),
    }


def _require_string(row: Mapping[str, Any], key: str, line_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest line {line_number}: {key} must be non-empty text")
    return value


def _require_integer(row: Mapping[str, Any], key: str, line_number: int) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"manifest line {line_number}: {key} must be an integer")
    return value


def _resolve_mixture_path(manifest_parent: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"mixture_path must be relative to the manifest: {relative}")
    resolved_parent = manifest_parent.resolve()
    resolved = (resolved_parent / candidate).resolve()
    try:
        resolved.relative_to(resolved_parent)
    except ValueError as error:
        raise ValueError(
            f"mixture_path escapes the dataset root: {relative}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _read_manifest_rows(manifest: Path) -> tuple[Path, list[Mapping[str, Any]]]:
    resolved = manifest.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: list[Mapping[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"manifest line {line_number}: blank rows are forbidden"
                )
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"manifest line {line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {resolved}")
    return resolved, rows


def _reject_symlink_components(
    path: Path,
    *,
    description: str,
    root: Path | None = None,
) -> None:
    """Reject symlinks without resolving away evidence that one was used."""

    if root is None:
        absolute = path.absolute()
        parts = absolute.parts
        current = Path(parts[0])
        remaining = parts[1:]
    else:
        current = root.resolve()
        try:
            relative = path.absolute().relative_to(current)
        except ValueError as error:
            raise ValueError(f"{description} escapes its dataset root") from error
        remaining = relative.parts
    for part in remaining:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{description} must not use symlinks: {current}")


def _resolve_real10_mixture_path(
    manifest: Path,
    record: Any,
) -> Path:
    """Use the canonical resolver, then fail closed on every symlink component."""

    resolved = resolve_manifest_mixture_path(manifest, record)
    root = manifest.resolve().parent
    lexical = root.joinpath(*Path(record.mixture_path).parts)
    _reject_symlink_components(
        lexical,
        description=f"QCES-Real-10 mixture {record.mixture_path!r}",
        root=root,
    )
    if not lexical.is_file():
        raise FileNotFoundError(lexical)
    if lexical.resolve() != resolved:
        raise RuntimeError("QCES-Real-10 mixture resolution changed unexpectedly")
    return resolved


def _validate_real10_manifest_path(
    manifest: Path,
    *,
    expected_resolved: Path | None = None,
) -> Path:
    _reject_symlink_components(
        manifest,
        description="QCES-Real-10 inference manifest",
    )
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    resolved = manifest.resolve()
    if expected_resolved is not None and resolved != expected_resolved:
        raise RuntimeError("QCES-Real-10 inference manifest path changed")
    return resolved


def _legacy_manifest_spec(
    resolved: Path,
    rows: Sequence[Mapping[str, Any]],
) -> CacheManifestSpec:
    """Preserve the original permissive-extra-field QCES-v5 cache contract."""

    samples: list[SampleSpec] = []
    scenes: dict[str, SceneSpec] = {}
    path_owners: dict[Path, str] = {}
    sample_ids: set[str] = set()
    schema_versions: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        sample_id = _require_string(row, "id", line_number)
        scene_id = _require_string(row, "scene_id", line_number)
        question = _require_string(row, "question", line_number)
        schema_versions.add(_require_string(row, "schema_version", line_number))
        mixture_relative = _require_string(row, "mixture_path", line_number)
        sample_rate = _require_integer(row, "sample_rate", line_number)
        num_samples = _require_integer(row, "num_samples", line_number)
        num_channels = _require_integer(row, "num_channels", line_number)
        duration = row.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(
                f"manifest line {line_number}: duration_seconds must be numeric"
            )
        duration = float(duration)
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample ID: {sample_id}")
        sample_ids.add(sample_id)
        if sample_rate != CANONICAL_SAMPLE_RATE:
            raise ValueError(
                f"{sample_id}: expected {CANONICAL_SAMPLE_RATE} Hz, got {sample_rate}"
            )
        if num_samples != CANONICAL_NUM_SAMPLES:
            raise ValueError(
                f"{sample_id}: expected {CANONICAL_NUM_SAMPLES} samples, "
                f"got {num_samples}"
            )
        if num_channels != 1:
            raise ValueError(f"{sample_id}: expected mono audio, got {num_channels}")
        if not math.isclose(
            duration,
            CANONICAL_DURATION_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{sample_id}: expected 10.0 s, got {duration}")
        mixture_path = _resolve_mixture_path(resolved.parent, mixture_relative)
        scene = SceneSpec(
            scene_id=scene_id,
            mixture_path=mixture_path,
            mixture_manifest_path=mixture_relative,
            sample_rate=sample_rate,
            num_samples=num_samples,
            duration_seconds=duration,
        )
        existing = scenes.get(scene_id)
        if existing is not None and existing != scene:
            raise ValueError(
                f"scene {scene_id} has inconsistent canonical mixture metadata"
            )
        owner = path_owners.get(mixture_path)
        if owner is not None and owner != scene_id:
            raise ValueError(
                f"mixture {mixture_relative} is ambiguously owned by {owner} and "
                f"{scene_id}"
            )
        scenes[scene_id] = scene
        path_owners[mixture_path] = scene_id
        samples.append(
            SampleSpec(sample_id=sample_id, scene_id=scene_id, question=question)
        )
    return CacheManifestSpec(
        samples=tuple(sorted(samples, key=lambda item: item.sample_id)),
        scenes=tuple(sorted(scenes.values(), key=lambda item: item.scene_id)),
        schema_versions=tuple(sorted(schema_versions)),
    )


def _real10_manifest_spec(
    manifest: Path,
    rows: Sequence[Mapping[str, Any]],
) -> CacheManifestSpec:
    _validate_real10_manifest_path(manifest)
    records = parse_inference_manifest(rows)
    scenes: dict[str, SceneSpec] = {}
    path_owners: dict[Path, str] = {}
    for record in records:
        mixture_path = _resolve_real10_mixture_path(manifest, record)
        scene = SceneSpec(
            scene_id=record.scene_id,
            mixture_path=mixture_path,
            mixture_manifest_path=record.mixture_path,
            sample_rate=record.sample_rate,
            num_samples=record.num_samples,
            duration_seconds=record.duration_seconds,
            declared_mixture_sha256=record.mixture_sha256,
            dataset_root=manifest.resolve().parent,
        )
        previous = scenes.setdefault(record.scene_id, scene)
        if previous != scene:
            raise ValueError(
                f"scene {record.scene_id} has inconsistent QCES-Real-10 audio"
            )
        owner = path_owners.setdefault(mixture_path, record.scene_id)
        if owner != record.scene_id:
            raise ValueError(
                f"mixture {record.mixture_path} is ambiguously owned by {owner} "
                f"and {record.scene_id}"
            )
    result = CacheManifestSpec(
        samples=tuple(
            sorted(
                (
                    SampleSpec(
                        sample_id=record.sample_id,
                        scene_id=record.scene_id,
                        question=record.question,
                    )
                    for record in records
                ),
                key=lambda item: item.sample_id,
            )
        ),
        scenes=tuple(sorted(scenes.values(), key=lambda item: item.scene_id)),
        schema_versions=(REAL10_INFERENCE_SCHEMA_VERSION,),
        real10_inference_manifest_fingerprint=(
            canonical_inference_manifest_fingerprint(records)
        ),
    )
    for scene in result.scenes:
        _validate_real10_scene_file(scene)
    return result


def read_cache_manifest(manifest: Path) -> CacheManifestSpec:
    """Dispatch Real-10 to its exact contract while retaining legacy behavior."""

    resolved, rows = _read_manifest_rows(manifest)
    schema_versions = {
        _require_string(row, "schema_version", line_number)
        for line_number, row in enumerate(rows, start=1)
    }
    real10_versions = {
        version for version in schema_versions if version.startswith("qces_real10_")
    }
    if real10_versions:
        if schema_versions != {REAL10_INFERENCE_SCHEMA_VERSION}:
            raise ValueError(
                "frozen-controller caching accepts only the exact "
                f"{REAL10_INFERENCE_SCHEMA_VERSION!r} model-input view; "
                "QCES-Real-10 scoring/oracle rows are forbidden "
                f"(got {sorted(schema_versions)!r})"
            )
        return _real10_manifest_spec(manifest, rows)
    return _legacy_manifest_spec(resolved, rows)


def read_strict_manifest(
    manifest: Path,
) -> tuple[list[SampleSpec], list[SceneSpec], list[str]]:
    """Read inference-safe cache fields under legacy or Real-10 contracts."""

    parsed = read_cache_manifest(manifest)
    return list(parsed.samples), list(parsed.scenes), list(parsed.schema_versions)


def configure_determinism(device: torch.device, seed: int) -> dict[str, Any]:
    """Configure deterministic execution, failing closed for CUDA prerequisites."""

    if device.type == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in VALID_CUBLAS_WORKSPACE_CONFIGS:
            allowed = ", ".join(sorted(VALID_CUBLAS_WORKSPACE_CONFIGS))
            raise RuntimeError(
                "deterministic CUDA caching requires CUBLAS_WORKSPACE_CONFIG to "
                f"already be one of {{{allowed}}}; got {workspace!r}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": seed,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def extract_query_encoder_state(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract the exact frozen CLAP state embedded in an AudioSep checkpoint."""

    state = {
        key.removeprefix("query_encoder."): value
        for key, value in payload.items()
        if isinstance(key, str) and key.startswith("query_encoder.")
    }
    if len(state) != EXPECTED_QUERY_STATE_TENSORS:
        raise RuntimeError(
            "unexpected AudioSep query-encoder state size: expected "
            f"{EXPECTED_QUERY_STATE_TENSORS}, got {len(state)}"
        )
    non_tensors = sorted(
        key for key, value in state.items() if not torch.is_tensor(value)
    )
    if non_tensors:
        raise RuntimeError(f"query-encoder state has non-tensors: {non_tensors}")
    return state


def load_query_encoder_state_exact(
    encoder: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Normalize only known nonpersistent buffers, then require an exact load."""

    expected_keys = set(encoder.state_dict())
    state_keys = set(state)
    unexpected_before_normalization = sorted(state_keys - expected_keys)
    ignored_nonpersistent = [
        key
        for key in unexpected_before_normalization
        if key.endswith(NONPERSISTENT_POSITION_ID_SUFFIX)
    ]
    unsupported_unexpected = sorted(
        set(unexpected_before_normalization) - set(ignored_nonpersistent)
    )
    if unsupported_unexpected:
        raise RuntimeError(
            "unexpected AudioSep CLAP state keys before load: "
            f"{unsupported_unexpected}"
        )
    load_state = {key: value for key, value in state.items() if key in expected_keys}
    missing_before_load = sorted(expected_keys - set(load_state))
    if missing_before_load:
        raise RuntimeError(
            f"missing AudioSep CLAP state keys before load: {missing_before_load}"
        )
    incompatible = encoder.load_state_dict(load_state, strict=True)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "strict AudioSep CLAP load was incompatible: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "extracted_tensor_count": len(state),
        "loaded_tensor_count": len(load_state),
        "effective_missing_keys": missing,
        "effective_unexpected_keys": unexpected,
        "ignored_nonpersistent_checkpoint_keys": ignored_nonpersistent,
    }


def load_frozen_encoder(
    audiosep_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    root = audiosep_root.resolve()
    checkpoint = checkpoint.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from models.clap_encoder import CLAP_Encoder

    encoder = CLAP_Encoder(pretrained_path="").eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise RuntimeError("AudioSep checkpoint must be a flat state mapping")
    state = extract_query_encoder_state(payload)
    provenance = load_query_encoder_state_exact(encoder, state)
    if getattr(encoder, "enable_fusion", None) is not False:
        raise RuntimeError("this cache format requires AudioSep CLAP fusion=False")
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    provenance["fusion_enabled"] = False
    del payload, state
    gc.collect()
    return encoder, provenance


def encode_full_questions(
    encoder: torch.nn.Module,
    samples: Sequence[SampleSpec],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Embed unmodified full questions, deduplicating text only for compute."""

    if batch_size <= 0:
        raise ValueError("text batch size must be positive")
    text_to_embedding: dict[str, torch.Tensor] = {}
    unique_texts = sorted({sample.question for sample in samples})
    with torch.inference_mode():
        for start in range(0, len(unique_texts), batch_size):
            texts = unique_texts[start : start + batch_size]
            embeddings = encoder.get_query_embed(modality="text", text=texts)
            if embeddings.shape != (len(texts), JOINT_DIM):
                raise RuntimeError(
                    "unexpected CLAP text feature shape: "
                    f"expected {(len(texts), JOINT_DIM)}, got {tuple(embeddings.shape)}"
                )
            embeddings = embeddings.detach().float().cpu().contiguous()
            if not torch.isfinite(embeddings).all():
                raise RuntimeError("CLAP text features contain non-finite values")
            norms = torch.linalg.vector_norm(embeddings, dim=-1)
            if not torch.allclose(
                norms,
                torch.ones_like(norms),
                rtol=1e-4,
                atol=1e-4,
            ):
                raise RuntimeError("CLAP text features are not unit normalized")
            text_to_embedding.update(dict(zip(texts, embeddings.unbind(0))))
    return {
        sample.sample_id: text_to_embedding[sample.question].clone()
        for sample in samples
    }


def compress_repeated_fine_features(fine: torch.Tensor) -> torch.Tensor:
    """Collapse 1024 exact repeated frames to the 32 effective CLAP frames."""

    expected = (fine.shape[0], RAW_FINE_FRAMES, FINE_DIM)
    if fine.ndim != 3 or tuple(fine.shape[1:]) != (RAW_FINE_FRAMES, FINE_DIM):
        raise RuntimeError(
            "unexpected raw fine-grained CLAP shape: expected "
            f"[B,{RAW_FINE_FRAMES},{FINE_DIM}], got {tuple(fine.shape)}"
        )
    grouped = fine.reshape(fine.shape[0], EFFECTIVE_FRAMES, REPEAT_RATIO, FINE_DIM)
    reference = grouped[:, :, :1, :]
    if not torch.equal(grouped, reference.expand_as(grouped)):
        max_difference = (grouped - reference).abs().max().item()
        raise RuntimeError(
            "raw fine-grained CLAP frames are not exact 32-repeat blocks; "
            f"max_abs_difference={max_difference}"
        )
    result = reference.squeeze(2)
    if result.shape != (fine.shape[0], EFFECTIVE_FRAMES, FINE_DIM):
        raise AssertionError(f"internal effective-frame shape failure from {expected}")
    return result


def project_effective_audio_features(
    fine: torch.Tensor,
    projection: torch.nn.Module,
) -> torch.Tensor:
    """Compress, project to the shared space, then normalize every frame."""

    effective = compress_repeated_fine_features(fine)
    projected = projection(effective)
    if projected.shape != (fine.shape[0], EFFECTIVE_FRAMES, JOINT_DIM):
        raise RuntimeError(
            "unexpected projected CLAP audio shape: expected "
            f"{(fine.shape[0], EFFECTIVE_FRAMES, JOINT_DIM)}, "
            f"got {tuple(projected.shape)}"
        )
    if not torch.isfinite(projected).all():
        raise RuntimeError("projected CLAP audio features contain non-finite values")
    norms = torch.linalg.vector_norm(projected.float(), dim=-1)
    if torch.any(norms <= 1e-12):
        raise RuntimeError("projected CLAP audio features contain a zero-norm frame")
    normalized = F.normalize(projected.float(), dim=-1)
    if not torch.isfinite(normalized).all():
        raise RuntimeError("normalized CLAP audio features contain non-finite values")
    return normalized


def _validate_real10_scene_file(scene: SceneSpec) -> dict[str, Any]:
    """Verify the current canonical WAV bytes and reject path substitution."""

    if scene.declared_mixture_sha256 is None or scene.dataset_root is None:
        raise TypeError(f"{scene.scene_id}: missing strict Real-10 scene identity")
    lexical = scene.dataset_root.joinpath(*Path(scene.mixture_manifest_path).parts)
    _reject_symlink_components(
        lexical,
        description=f"QCES-Real-10 mixture {scene.mixture_manifest_path!r}",
        root=scene.dataset_root,
    )
    if not lexical.is_file() or lexical.resolve() != scene.mixture_path:
        raise RuntimeError(f"{scene.scene_id}: canonical mixture path changed")
    identity = file_identity(lexical)
    if identity["sha256"] != scene.declared_mixture_sha256:
        raise RuntimeError(
            f"{scene.scene_id}: mixture_sha256 does not match canonical WAV bytes"
        )
    try:
        info = sf.info(str(lexical))
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"{scene.scene_id}: cannot inspect canonical QCES-Real-10 WAV"
        ) from error
    if (
        info.format != REAL10_CANONICAL_WAV_FORMAT
        or info.subtype != REAL10_CANONICAL_WAV_SUBTYPE
        or info.samplerate != CANONICAL_SAMPLE_RATE
        or info.channels != 1
        or info.frames != CANONICAL_NUM_SAMPLES
    ):
        raise RuntimeError(
            f"{scene.scene_id}: canonical mixture must be mono 32 kHz/10 s "
            "IEEE-float WAV"
        )
    return identity


def _read_real10_waveform(scene: SceneSpec) -> np.ndarray:
    """Hash and decode the same immutable byte buffer, closing the TOCTOU gap."""

    before = _validate_real10_scene_file(scene)
    with scene.mixture_path.open("rb") as handle:
        wav_bytes = handle.read()
    byte_sha256 = hashlib.sha256(wav_bytes).hexdigest()
    if byte_sha256 != scene.declared_mixture_sha256:
        raise RuntimeError(
            f"{scene.scene_id}: canonical WAV mutated while its bytes were read"
        )
    try:
        info = sf.info(io.BytesIO(wav_bytes))
        waveform, sample_rate = sf.read(
            io.BytesIO(wav_bytes),
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"{scene.scene_id}: cannot decode canonical QCES-Real-10 WAV"
        ) from error
    if (
        info.format != REAL10_CANONICAL_WAV_FORMAT
        or info.subtype != REAL10_CANONICAL_WAV_SUBTYPE
        or info.samplerate != CANONICAL_SAMPLE_RATE
        or info.channels != 1
        or info.frames != CANONICAL_NUM_SAMPLES
    ):
        raise RuntimeError(
            f"{scene.scene_id}: expected mono 32 kHz/10 s IEEE-float WAV, got "
            f"format={info.format}, subtype={info.subtype}, "
            f"rate={info.samplerate}, channels={info.channels}, frames={info.frames}"
        )
    after = _validate_real10_scene_file(scene)
    _assert_identity_unchanged(
        f"canonical mixture for {scene.scene_id}",
        before,
        after,
    )
    if sample_rate != info.samplerate:
        raise RuntimeError(f"{scene.scene_id}: WAV decoder changed the sample rate")
    return waveform


def read_canonical_waveform(scene: SceneSpec) -> np.ndarray:
    if scene.declared_mixture_sha256 is not None or scene.dataset_root is not None:
        waveform = _read_real10_waveform(scene)
        sample_rate = CANONICAL_SAMPLE_RATE
    else:
        waveform, sample_rate = sf.read(
            scene.mixture_path,
            dtype="float32",
            always_2d=True,
        )
    if sample_rate != CANONICAL_SAMPLE_RATE:
        raise RuntimeError(
            f"{scene.scene_id}: WAV rate {sample_rate} != {CANONICAL_SAMPLE_RATE}"
        )
    if waveform.shape != (CANONICAL_NUM_SAMPLES, 1):
        raise RuntimeError(
            f"{scene.scene_id}: WAV shape {waveform.shape} != "
            f"({CANONICAL_NUM_SAMPLES}, 1)"
        )
    if not np.isfinite(waveform).all():
        raise RuntimeError(f"{scene.scene_id}: WAV contains non-finite samples")
    return waveform[:, 0]


def encode_canonical_waveform(
    encoder: torch.nn.Module,
    waveform: np.ndarray,
    *,
    scene_id: str,
    device: torch.device,
) -> torch.Tensor:
    """Encode one canonical waveform without requiring manifest metadata.

    This is shared by the sealed offline-cache path and online QCES inference.
    Keeping both routes on the same implementation prevents a silent feature
    mismatch between validation caches and user-supplied 10-second clips.
    """

    import torchaudio

    if not isinstance(waveform, np.ndarray):
        raise TypeError("canonical waveform must be a NumPy array")
    if waveform.shape != (CANONICAL_NUM_SAMPLES,):
        raise RuntimeError(
            f"{scene_id}: waveform shape {waveform.shape} != "
            f"({CANONICAL_NUM_SAMPLES},)"
        )
    if waveform.dtype != np.float32:
        raise RuntimeError(f"{scene_id}: canonical waveform must be float32")
    if not np.isfinite(waveform).all():
        raise RuntimeError(f"{scene_id}: waveform contains non-finite samples")
    waveform_tensor = torch.from_numpy(np.ascontiguousarray(waveform)).to(device)
    waveform_tensor = torchaudio.functional.resample(
        waveform_tensor,
        orig_freq=CANONICAL_SAMPLE_RATE,
        new_freq=CLAP_SAMPLE_RATE,
    )
    if waveform_tensor.shape != (CLAP_NUM_SAMPLES,):
        raise RuntimeError(
            f"{scene_id}: resampled shape {tuple(waveform_tensor.shape)} != "
            f"({CLAP_NUM_SAMPLES},)"
        )
    audio_input = {
        "waveform": waveform_tensor.unsqueeze(0),
        "longer": torch.zeros((1, 1), dtype=torch.bool, device=device),
    }
    with torch.inference_mode():
        raw = encoder.model.encode_audio(audio_input, device=device)
        if "fine_grained_embedding" not in raw:
            raise RuntimeError("CLAP encode_audio omitted fine_grained_embedding")
        normalized = project_effective_audio_features(
            raw["fine_grained_embedding"],
            encoder.model.audio_projection,
        )
    return normalized.squeeze(0).half().cpu().contiguous()


def encode_scene(
    encoder: torch.nn.Module,
    scene: SceneSpec,
    device: torch.device,
) -> torch.Tensor:
    return encode_canonical_waveform(
        encoder,
        read_canonical_waveform(scene),
        scene_id=scene.scene_id,
        device=device,
    )


def question_payload(features: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    payload = {
        "format": QUESTION_CACHE_FORMAT,
        "feature_space": FEATURE_SPACE,
        "dtype": "float32",
        "shape": [JOINT_DIM],
        "features": dict(features),
    }
    validate_question_payload(payload)
    return payload


def scene_payload(features: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    payload = {
        "format": SCENE_CACHE_FORMAT,
        "feature_space": FEATURE_SPACE,
        "dtype": "float16",
        "shape": [EFFECTIVE_FRAMES, JOINT_DIM],
        "raw_fine_frames": RAW_FINE_FRAMES,
        "repeat_ratio": REPEAT_RATIO,
        "features": dict(features),
    }
    validate_scene_payload(payload)
    return payload


def _validate_feature_mapping(
    features: Any,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if not isinstance(features, Mapping) or not features:
        raise ValueError("features must be a non-empty mapping")
    for identifier, tensor in features.items():
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("feature IDs must be non-empty strings")
        if not torch.is_tensor(tensor):
            raise TypeError(f"feature {identifier} is not a tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"feature {identifier} shape {tuple(tensor.shape)} != {shape}"
            )
        if tensor.dtype != dtype:
            raise ValueError(f"feature {identifier} dtype {tensor.dtype} != {dtype}")
        if tensor.device.type != "cpu":
            raise ValueError(f"feature {identifier} must be stored on CPU")
        if not tensor.is_contiguous():
            raise ValueError(f"feature {identifier} must be contiguous")
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"feature {identifier} has non-finite values")


def validate_question_payload(payload: Mapping[str, Any]) -> None:
    allowed = frozenset({"format", "feature_space", "dtype", "shape", "features"})
    if set(payload) != allowed:
        raise ValueError(
            "unexpected question-cache fields: " f"{sorted(set(payload) - allowed)}"
        )
    if payload["format"] != QUESTION_CACHE_FORMAT:
        raise ValueError("wrong question-cache format")
    if payload["feature_space"] != FEATURE_SPACE:
        raise ValueError("wrong question feature space")
    if payload["dtype"] != "float32" or payload["shape"] != [JOINT_DIM]:
        raise ValueError("wrong question-cache dtype or shape declaration")
    _validate_feature_mapping(
        payload["features"], shape=(JOINT_DIM,), dtype=torch.float32
    )


def validate_scene_payload(payload: Mapping[str, Any]) -> None:
    allowed = frozenset(
        {
            "format",
            "feature_space",
            "dtype",
            "shape",
            "raw_fine_frames",
            "repeat_ratio",
            "features",
        }
    )
    if set(payload) != allowed:
        raise ValueError(
            f"unexpected scene-cache fields: {sorted(set(payload) - allowed)}"
        )
    if payload["format"] != SCENE_CACHE_FORMAT:
        raise ValueError("wrong scene-cache format")
    if payload["feature_space"] != FEATURE_SPACE:
        raise ValueError("wrong scene feature space")
    if payload["dtype"] != "float16" or payload["shape"] != [
        EFFECTIVE_FRAMES,
        JOINT_DIM,
    ]:
        raise ValueError("wrong scene-cache dtype or shape declaration")
    if payload["raw_fine_frames"] != RAW_FINE_FRAMES:
        raise ValueError("wrong raw fine-frame count")
    if payload["repeat_ratio"] != REPEAT_RATIO:
        raise ValueError("wrong repeat ratio")
    _validate_feature_mapping(
        payload["features"],
        shape=(EFFECTIVE_FRAMES, JOINT_DIM),
        dtype=torch.float16,
    )
    for identifier, tensor in payload["features"].items():
        norms = torch.linalg.vector_norm(tensor.float(), dim=-1)
        if not torch.allclose(
            norms,
            torch.ones_like(norms),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError(f"scene feature {identifier} is not frame-L2-normalized")


def _software_identity() -> dict[str, str]:
    packages = ("numpy", "soundfile", "torch", "torchaudio", "transformers")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {"python": platform.python_version(), **versions}


def _aggregate_mixture_hash(identities: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        for key in ("scene_id", "manifest_path", "sha256", "size_bytes"):
            encoded = str(identity[key]).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def real10_receipt_contract(manifest: CacheManifestSpec) -> dict[str, Any]:
    """Materialize the label-free Real-10 binding stored in a cache receipt."""

    if not manifest.is_real10_inference:
        raise ValueError("receipt contract requires a QCES-Real-10 inference manifest")
    declared_hashes = {
        scene.scene_id: scene.declared_mixture_sha256 for scene in manifest.scenes
    }
    if not declared_hashes or any(value is None for value in declared_hashes.values()):
        raise ValueError("strict QCES-Real-10 scene lacks a declared WAV SHA256")
    fingerprint = manifest.real10_inference_manifest_fingerprint
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("strict QCES-Real-10 manifest lacks its canonical fingerprint")
    return {
        "schema_version": REAL10_INFERENCE_SCHEMA_VERSION,
        "canonical_inference_manifest_fingerprint": fingerprint,
        "canonical_wav_sha256_by_scene": declared_hashes,
        "contains_event_answer_or_oracle_inputs": False,
    }


def _assert_identity_unchanged(
    description: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    if dict(before) != dict(after):
        raise RuntimeError(
            f"{description} changed while it was being consumed; refusing to "
            "write an ambiguous cache receipt"
        )


def write_outputs_atomically(
    output_dir: Path,
    questions: Mapping[str, Any],
    scenes: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Path:
    """Write a new cache directory without overwriting or leaving partial output."""

    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        question_file = staging / "question_features.pt"
        scene_file = staging / "scene_audio_features.pt"
        torch.save(dict(questions), question_file)
        torch.save(dict(scenes), scene_file)
        full_receipt = {
            **dict(receipt),
            "artifacts": {
                "question_features": {
                    "filename": question_file.name,
                    "sha256": sha256_file(question_file),
                    "size_bytes": question_file.stat().st_size,
                },
                "scene_audio_features": {
                    "filename": scene_file.name,
                    "sha256": sha256_file(scene_file),
                    "size_bytes": scene_file.stat().st_size,
                },
            },
        }
        receipt_file = staging / "cache_receipt.json"
        receipt_file.write_text(
            json.dumps(full_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.text_batch_size <= 0:
        raise SystemExit("--text-batch-size must be positive")
    manifest_input = args.manifest.absolute()
    manifest = manifest_input.resolve()
    audiosep_root = args.audiosep_root.resolve()
    checkpoint = args.audiosep_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_dir}")

    manifest_identity = file_identity(manifest_input)
    manifest_spec = read_cache_manifest(manifest_input)
    samples = list(manifest_spec.samples)
    scenes = list(manifest_spec.scenes)
    schema_versions = list(manifest_spec.schema_versions)
    _assert_identity_unchanged("manifest", manifest_identity, file_identity(manifest))
    checkpoint_identity = file_identity(checkpoint)
    source_identity = source_tree_identity(audiosep_root)
    device = torch.device(args.device)
    deterministic = configure_determinism(device, args.seed)
    encoder, state_provenance = load_frozen_encoder(
        audiosep_root,
        checkpoint,
        device,
    )
    _assert_identity_unchanged(
        "AudioSep checkpoint", checkpoint_identity, file_identity(checkpoint)
    )
    _assert_identity_unchanged(
        "AudioSep source tree", source_identity, source_tree_identity(audiosep_root)
    )

    question_features = encode_full_questions(
        encoder,
        samples,
        args.text_batch_size,
    )
    scene_features: dict[str, torch.Tensor] = {}
    mixture_identities: list[dict[str, Any]] = []
    mixture_identity_by_scene: dict[str, dict[str, Any]] = {}
    for index, scene in enumerate(scenes, start=1):
        identity = (
            _validate_real10_scene_file(scene)
            if manifest_spec.is_real10_inference
            else file_identity(scene.mixture_path)
        )
        scene_features[scene.scene_id] = encode_scene(encoder, scene, device)
        identity_after = (
            _validate_real10_scene_file(scene)
            if manifest_spec.is_real10_inference
            else file_identity(scene.mixture_path)
        )
        _assert_identity_unchanged(
            f"mixture for {scene.scene_id}",
            identity,
            identity_after,
        )
        mixture_identity = {
            "scene_id": scene.scene_id,
            "manifest_path": scene.mixture_manifest_path,
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
            "sample_rate": scene.sample_rate,
            "num_samples": scene.num_samples,
            "duration_seconds": scene.duration_seconds,
            "feature_shape": [EFFECTIVE_FRAMES, JOINT_DIM],
            "feature_dtype": "float16",
        }
        if scene.declared_mixture_sha256 is not None:
            mixture_identity["declared_mixture_sha256"] = scene.declared_mixture_sha256
        mixture_identities.append(mixture_identity)
        mixture_identity_by_scene[scene.scene_id] = identity
        print(f"cached scene {index}/{len(scenes)}: {scene.scene_id}", flush=True)

    if manifest_spec.is_real10_inference:
        _validate_real10_manifest_path(
            manifest_input,
            expected_resolved=manifest,
        )
    _assert_identity_unchanged("manifest", manifest_identity, file_identity(manifest))
    for scene in scenes:
        current_identity = (
            _validate_real10_scene_file(scene)
            if manifest_spec.is_real10_inference
            else file_identity(scene.mixture_path)
        )
        _assert_identity_unchanged(
            f"mixture for {scene.scene_id}",
            mixture_identity_by_scene[scene.scene_id],
            current_identity,
        )

    questions = question_payload(question_features)
    scene_audio = scene_payload(scene_features)
    sample_ids = sorted(question_features)
    scene_ids = sorted(scene_features)
    receipt = {
        "format": RECEIPT_FORMAT,
        "purpose": "frozen_controller_inputs_without_oracle_labels",
        "manifest": manifest_identity,
        "schema_versions": schema_versions,
        "audiosep_checkpoint": checkpoint_identity,
        "audiosep_source_tree": source_identity,
        "query_encoder_state": state_provenance,
        "canonical_audio": {
            "sample_rate": CANONICAL_SAMPLE_RATE,
            "num_samples": CANONICAL_NUM_SAMPLES,
            "duration_seconds": CANONICAL_DURATION_SECONDS,
            "num_channels": 1,
            "clap_sample_rate": CLAP_SAMPLE_RATE,
            "clap_num_samples": CLAP_NUM_SAMPLES,
        },
        "features": {
            "feature_space": FEATURE_SPACE,
            "question_shape": [JOINT_DIM],
            "question_dtype": "float32",
            "raw_audio_shape": [RAW_FINE_FRAMES, FINE_DIM],
            "effective_audio_shape": [EFFECTIVE_FRAMES, JOINT_DIM],
            "effective_audio_dtype": "float16",
            "raw_to_effective_repeat_ratio": REPEAT_RATIO,
            "audio_normalization": "per_frame_l2_before_fp16_storage",
        },
        "counts": {
            "sample_ids": len(sample_ids),
            "scene_ids": len(scene_ids),
            "unique_full_question_texts": len({sample.question for sample in samples}),
            "physical_mixture_encodes": len(scene_ids),
        },
        "sample_ids": sample_ids,
        "scene_ids": scene_ids,
        "mixtures": mixture_identities,
        "mixtures_aggregate_sha256": _aggregate_mixture_hash(mixture_identities),
        "execution": {
            "device": str(device),
            "text_batch_size": args.text_batch_size,
            "determinism": deterministic,
            "software": _software_identity(),
        },
        "privacy_contract": {
            "question_text_stored": False,
            "event_labels_stored": False,
            "answer_labels_stored": False,
            "evidence_annotations_stored": False,
            "allowed_keys": "sample_id_and_scene_id_only",
        },
    }
    if manifest_spec.is_real10_inference:
        receipt["qces_real10_inference_contract"] = real10_receipt_contract(
            manifest_spec
        )
        receipt["privacy_contract"]["contains_event_answer_or_oracle_inputs"] = False
    output = write_outputs_atomically(output_dir, questions, scene_audio, receipt)
    print(f"wrote strict frozen-controller cache: {output}")

    del encoder, question_features, scene_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
