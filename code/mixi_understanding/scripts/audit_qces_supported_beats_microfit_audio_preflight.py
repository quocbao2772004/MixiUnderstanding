#!/usr/bin/env python3
"""Fail-closed audio preflight for the 200-class BEATs-v2 microfit.

The metadata gate and this preflight intentionally have disjoint jobs.  The
metadata gate binds the manifests, ontology, split identities, and native head.
This sidecar then proves that the *same hashed manifests* point at decodable,
non-pathological waveforms whose byte hashes and decoded geometry agree with
their declarations.  It performs no model import, GPU work, or training.

Only a passing receipt from this program may set ``audio_training_authorized``.
The authorization is debug-microfit-only and is hard-bound to 200--500 train
scenes and the fixed 250 x 40-ms detector grid.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FORMAT = "qces_supported_beats_microfit_audio_preflight_v1"
METADATA_GATE_FORMAT = "qces_supported_beats_microfit_metadata_gate_v1"
TRAIN_SCENES_MIN = 200
TRAIN_SCENES_MAX = 500
GRID_FRAMES = 250
FRAME_HOP_SECONDS = 0.04
FIXED_AUDIO_SECONDS = GRID_FRAMES * FRAME_HOP_SECONDS
EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 1
DEFAULT_CLIP_THRESHOLD = 0.999
DEFAULT_MINIMUM_RMS = 1e-5
DEFAULT_RECONSTRUCTION_TOLERANCE = 2e-6
MAX_FAILURES_IN_RECEIPT = 200
CANONICAL_SPLITS = ("train", "dev", "test")
STRICT_IDENTITY_FIELDS = (
    "scene_id",
    "video_id",
    "audio_sha256",
    "mixture_path",
    "source_id",
    "event.event_id",
    "event.source_id",
    "event.source_sha256",
    "event.source_path",
)


@dataclass(frozen=True)
class AudioAsset:
    path: Path
    sha256: str
    sample_rate: int
    frames: int
    channels: int
    duration_seconds: float
    peak: float
    rms: float
    clipped_samples: int
    samples: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON receipt must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _valid_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _split_name(row: Mapping[str, Any]) -> str:
    return str(row.get("protocol_split") or row.get("split") or "").strip()


def _identity_values(row: Mapping[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for field in ("scene_id", "video_id", "audio_sha256", "mixture_path", "source_id"):
        value = str(row.get(field) or "").strip()
        if value:
            values[field].add(value)
    for event in row.get("events") or ():
        for field in ("event_id", "source_id", "source_sha256", "source_path"):
            value = str(event.get(field) or "").strip()
            if value:
                values[f"event.{field}"].add(value)
    return dict(values)


def _audit_strict_identity_overlap(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    identities: dict[str, dict[str, set[str]]] = {}
    for split, rows in rows_by_split.items():
        collected: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            for field, values in _identity_values(row).items():
                collected[field].update(values)
        identities[split] = dict(collected)
    pairs: dict[str, Any] = {}
    total = 0
    split_names = [split for split in CANONICAL_SPLITS if split in rows_by_split]
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            fields: dict[str, Any] = {}
            for field in STRICT_IDENTITY_FIELDS:
                overlap = identities[left].get(field, set()) & identities[right].get(
                    field, set()
                )
                if overlap:
                    total += len(overlap)
                    fields[field] = {
                        "count": len(overlap),
                        "sample": sorted(overlap)[:10],
                    }
            pairs[f"{left}__{right}"] = {
                "overlap_count": sum(item["count"] for item in fields.values()),
                "fields": fields,
            }
    return {
        "identity_fields": list(STRICT_IDENTITY_FIELDS),
        "pairs": pairs,
        "overlap_count": total,
        "passes": total == 0,
    }


def _declared_frames(row: Mapping[str, Any], *, context: str) -> int:
    fields = (
        "audio_num_frames",
        "mixture_num_frames",
        "audio_num_samples",
        "num_samples",
    )
    present = [(field, row.get(field)) for field in fields if row.get(field) is not None]
    if not present:
        raise ValueError(
            f"{context} does not declare audio_num_frames; frame count may not be inferred"
        )
    values = {int(value) for _, value in present}
    if len(values) != 1 or next(iter(values)) <= 0:
        raise ValueError(f"{context} has inconsistent/invalid frame declarations: {present}")
    return next(iter(values))


def _declared_channels(row: Mapping[str, Any], *, context: str) -> int:
    fields = ("audio_num_channels", "mixture_num_channels", "num_channels")
    present = [(field, row.get(field)) for field in fields if row.get(field) is not None]
    if not present:
        raise ValueError(
            f"{context} does not declare audio_num_channels; channel count may not be inferred"
        )
    values = {int(value) for _, value in present}
    if len(values) != 1 or next(iter(values)) <= 0:
        raise ValueError(f"{context} has inconsistent/invalid channel declarations: {present}")
    return next(iter(values))


class OnceAudioReader:
    """Open every filesystem audio path exactly once and retain only its audit."""

    def __init__(
        self,
        *,
        audio_root: Path,
        clipping_threshold: float,
        minimum_rms: float,
    ) -> None:
        self.audio_root = audio_root.resolve(strict=True)
        if not self.audio_root.is_dir():
            raise NotADirectoryError(self.audio_root)
        self.clipping_threshold = float(clipping_threshold)
        self.minimum_rms = float(minimum_rms)
        self._opened: set[Path] = set()
        self.inventory: list[dict[str, Any]] = []
        self.role_counts: Counter[str] = Counter()

    def resolve(self, raw_path: object, *, context: str) -> Path:
        text = str(raw_path or "").strip()
        if not text:
            raise ValueError(f"{context} has an empty audio path")
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.audio_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{context} audio does not exist: {candidate}") from exc
        try:
            resolved.relative_to(self.audio_root)
        except ValueError as exc:
            raise ValueError(
                f"{context} resolves outside audio-root: {resolved} not under {self.audio_root}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"{context} is not a regular file: {resolved}")
        return resolved

    def read(
        self,
        raw_path: object,
        *,
        expected_sha256: object,
        role: str,
        context: str,
        require_audible: bool,
    ) -> AudioAsset:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("audio preflight requires soundfile") from exc

        resolved = self.resolve(raw_path, context=context)
        if resolved in self._opened:
            raise ValueError(
                f"audio path is referenced more than once; one-open contract violated: {resolved}"
            )
        expected = str(expected_sha256 or "").lower()
        if not _valid_sha256(expected):
            raise ValueError(f"{context} has no valid declared SHA-256")

        # A single filesystem open yields the exact encoded bytes.  Decoding is
        # then performed from that in-memory object, so hash and waveform cannot
        # race against a second path open.
        with resolved.open("rb") as handle:
            encoded = handle.read()
        self._opened.add(resolved)
        actual_sha = _sha256_bytes(encoded)
        if actual_sha != expected:
            raise ValueError(
                f"{context} SHA-256 mismatch: declared={expected} actual={actual_sha}"
            )

        with sf.SoundFile(io.BytesIO(encoded), mode="r") as handle:
            sample_rate = int(handle.samplerate)
            frames = int(handle.frames)
            channels = int(handle.channels)
            samples = handle.read(dtype="float32", always_2d=True)
        if samples.shape != (frames, channels):
            raise ValueError(
                f"{context} decoder shape mismatch: header={(frames, channels)} "
                f"decoded={samples.shape}"
            )
        if frames <= 0 or channels <= 0 or samples.size <= 0:
            raise ValueError(f"{context} decodes to empty audio")
        finite = np.isfinite(samples)
        if not bool(finite.all()):
            raise ValueError(f"{context} contains NaN or Inf samples")
        absolute = np.abs(samples)
        peak = float(np.max(absolute))
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
        clipped = int(np.count_nonzero(absolute >= self.clipping_threshold))
        if peak > 1.0 + 1e-7:
            raise ValueError(f"{context} contains out-of-range samples: peak={peak:.9f}")
        if clipped:
            raise ValueError(
                f"{context} contains {clipped} samples at/above clipping threshold "
                f"{self.clipping_threshold}"
            )
        if require_audible and rms < self.minimum_rms:
            raise ValueError(
                f"{context} is silent/near-silent: rms={rms:.9g} < {self.minimum_rms}"
            )

        duration = frames / float(sample_rate)
        self.role_counts[role] += 1
        self.inventory.append(
            {
                "path": resolved.as_posix(),
                "sha256": actual_sha,
                "role": role,
                "sample_rate": sample_rate,
                "frames": frames,
                "channels": channels,
            }
        )
        return AudioAsset(
            path=resolved,
            sha256=actual_sha,
            sample_rate=sample_rate,
            frames=frames,
            channels=channels,
            duration_seconds=duration,
            peak=peak,
            rms=rms,
            clipped_samples=clipped,
            samples=samples,
        )

    @property
    def files_opened(self) -> int:
        return len(self._opened)


def _metadata_gate_failures(
    gate: Mapping[str, Any],
    *,
    manifests: Mapping[str, Path],
) -> list[str]:
    failures: list[str] = []
    if gate.get("format") != METADATA_GATE_FORMAT:
        failures.append(
            f"metadata gate format must be {METADATA_GATE_FORMAT}, got {gate.get('format')!r}"
        )
    scope = gate.get("scope") or {}
    if gate.get("passes") is not True or scope.get("metadata_training_authorized") is not True:
        failures.append("metadata gate does not authorize metadata training")
    if scope.get("debug_microfit_only") is not True or scope.get("paper_eligible") is not False:
        failures.append("metadata gate is not explicitly scoped to a non-paper debug microfit")

    contract = gate.get("contract") or {}
    if contract.get("train_scene_range") != [TRAIN_SCENES_MIN, TRAIN_SCENES_MAX]:
        failures.append(
            f"metadata gate train_scene_range must be "
            f"[{TRAIN_SCENES_MIN},{TRAIN_SCENES_MAX}]"
        )
    if int(contract.get("fixed_grid_frames", -1)) != GRID_FRAMES:
        failures.append(f"metadata gate does not bind the {GRID_FRAMES}-frame grid")
    if int(contract.get("boundary_dilation_frames", -1)) != 0:
        failures.append("metadata gate boundary dilation must be zero")
    if contract.get("test_used_for_selection") is not False:
        failures.append("metadata gate does not prove test_used_for_selection=false")

    integrity = gate.get("integrity") or {}
    if integrity.get("passes") is not True or int(integrity.get("overlap_count", -1)) != 0:
        failures.append("metadata gate does not prove zero train/dev/test identity overlap")
    cal_identity = ((gate.get("calibration_selection") or {}).get("identity_audit") or {})
    if cal_identity.get("passes") is not True or int(cal_identity.get("overlap_count", -1)) != 0:
        failures.append("metadata gate does not prove calibration/selection identity isolation")

    bound = (gate.get("inputs") or {}).get("manifests") or {}
    for split, path in manifests.items():
        entry = bound.get(split) or {}
        try:
            gate_path = Path(str(entry.get("path") or "")).resolve(strict=True)
        except (FileNotFoundError, OSError):
            gate_path = None
        if gate_path != path:
            failures.append(f"metadata gate path mismatch for {split}")
        actual_hash = _sha256(path)
        if entry.get("sha256") != actual_hash:
            failures.append(f"metadata gate hash mismatch for {split}")
    return failures


def _validate_manifest_identity(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    global_scene_ids: dict[str, str] = {}
    declared_mixture_paths: dict[str, str] = {}
    declared_mixture_hashes: dict[str, str] = {}
    event_ids: dict[str, str] = {}
    source_values: dict[str, dict[str, str]] = {
        field: {}
        for field in ("source_id", "source_video_id", "source_sha256", "source_path")
    }
    events = 0
    for split, rows in rows_by_split.items():
        for row_index, row in enumerate(rows):
            scene_id = str(row.get("scene_id") or "").strip()
            context = f"{split}[{row_index}]"
            if not scene_id:
                failures.append(f"{context} has no scene_id")
                continue
            if scene_id in global_scene_ids:
                failures.append(f"duplicate scene_id {scene_id}")
            else:
                global_scene_ids[scene_id] = split
            declared_split = _split_name(row)
            if declared_split != split:
                failures.append(
                    f"scene {scene_id} declares split={declared_split!r}, expected {split!r}"
                )
            mixture_path = str(row.get("mixture_path") or "").strip()
            mixture_sha = str(row.get("audio_sha256") or "").lower()
            if not mixture_path:
                failures.append(f"scene {scene_id} has no mixture_path")
            elif mixture_path in declared_mixture_paths:
                failures.append(
                    f"mixture_path reused by {declared_mixture_paths[mixture_path]} and {scene_id}"
                )
            else:
                declared_mixture_paths[mixture_path] = scene_id
            if not _valid_sha256(mixture_sha):
                failures.append(f"scene {scene_id} has no valid audio_sha256")
            elif mixture_sha in declared_mixture_hashes:
                failures.append(
                    f"mixture audio hash reused by {declared_mixture_hashes[mixture_sha]} and {scene_id}"
                )
            else:
                declared_mixture_hashes[mixture_sha] = scene_id

            for event_index, event in enumerate(row.get("events") or ()):
                events += 1
                event_context = f"{scene_id}.events[{event_index}]"
                event_id = str(event.get("event_id") or "").strip()
                if not event_id:
                    failures.append(f"{event_context} has no event_id")
                elif event_id in event_ids:
                    failures.append(
                        f"event_id reused by {event_ids[event_id]} and {event_context}"
                    )
                else:
                    event_ids[event_id] = event_context
                for field, registry in source_values.items():
                    value = str(event.get(field) or "").strip()
                    if not value:
                        failures.append(f"{event_context} has no {field}")
                        continue
                    if field == "source_sha256" and not _valid_sha256(value):
                        failures.append(f"{event_context} has invalid source_sha256")
                        continue
                    if value in registry:
                        failures.append(
                            f"{field} reused by {registry[value]} and {event_context}"
                        )
                    else:
                        registry[value] = event_context

    overlap = _audit_strict_identity_overlap(rows_by_split)
    if overlap.get("passes") is not True or int(overlap.get("overlap_count", -1)) != 0:
        failures.append("independent identity audit found train/dev/test overlap")
    return failures, {
        "passes": not failures,
        "scenes": len(global_scene_ids),
        "events": events,
        "unique_event_ids": len(event_ids),
        "unique_declared_mixture_paths": len(declared_mixture_paths),
        "unique_declared_mixture_hashes": len(declared_mixture_hashes),
        "unique_source_identities": {
            field: len(values) for field, values in source_values.items()
        },
        "cross_split": overlap,
    }


def _validate_resolved_audio_paths(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reader: OnceAudioReader,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    owners: dict[Path, str] = {}
    mixtures: list[dict[str, str]] = []

    def register(raw_path: object, *, context: str, role: str) -> None:
        if raw_path in (None, ""):
            return
        try:
            resolved = reader.resolve(raw_path, context=context)
        except (FileNotFoundError, OSError, ValueError) as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            return
        if resolved in owners:
            failures.append(
                f"resolved audio path reused by {owners[resolved]} and {context}: {resolved}"
            )
            return
        owners[resolved] = context
        if role == "mixture":
            mixtures.append({"context": context, "path": resolved.as_posix()})

    for split, rows in rows_by_split.items():
        for row_index, row in enumerate(rows):
            scene_id = str(row.get("scene_id") or f"{split}[{row_index}]")
            register(
                row.get("mixture_path"),
                context=f"scene {scene_id}.mixture",
                role="mixture",
            )
            for event_index, event in enumerate(row.get("events") or ()):
                register(
                    event.get("component_path"),
                    context=f"{scene_id}.events[{event_index}].component",
                    role="source_component",
                )
            for name, evidence_path, _, residual_path, _ in _evidence_contracts(row):
                register(
                    row.get(evidence_path),
                    context=f"scene {scene_id}.{name}",
                    role=name,
                )
                register(
                    row.get(residual_path),
                    context=f"scene {scene_id}.{name}_residual",
                    role=f"{name}_residual",
                )
    return failures, {
        "passes": not failures,
        "unique_audio_paths": len(owners),
        "resolved_mixture_paths": len(mixtures),
        "resolved_mixture_inventory_sha256": _canonical_json_sha256(
            sorted(mixtures, key=lambda item: (item["path"], item["context"]))
        ),
    }


def _grid_frame(value: float, *, context: str) -> int:
    if not math.isfinite(value):
        raise ValueError(f"{context} is not finite")
    scaled = value / FRAME_HOP_SECONDS
    rounded = int(round(scaled))
    if not math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(
            f"{context}={value} is not aligned to the {FRAME_HOP_SECONDS}-second grid"
        )
    if not 0 <= rounded <= GRID_FRAMES:
        raise ValueError(f"{context} maps outside fixed grid: frame={rounded}")
    return rounded


def _verify_mixture_declarations(
    row: Mapping[str, Any], asset: AudioAsset, *, scene_id: str
) -> None:
    try:
        declared_rate = int(row["sample_rate"])
        declared_duration = float(row["duration_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"scene {scene_id} must explicitly declare sample_rate and duration_seconds"
        ) from exc
    declared_frames = _declared_frames(row, context=f"scene {scene_id}")
    declared_channels = _declared_channels(row, context=f"scene {scene_id}")
    if declared_rate != EXPECTED_SAMPLE_RATE or asset.sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"scene {scene_id} sample rate must be {EXPECTED_SAMPLE_RATE}: "
            f"declared={declared_rate}, decoded={asset.sample_rate}"
        )
    if declared_channels != EXPECTED_CHANNELS or asset.channels != EXPECTED_CHANNELS:
        raise ValueError(
            f"scene {scene_id} must be mono: declared={declared_channels}, "
            f"decoded={asset.channels}"
        )
    if declared_frames != asset.frames:
        raise ValueError(
            f"scene {scene_id} frame-count mismatch: declared={declared_frames}, "
            f"decoded={asset.frames}"
        )
    if not math.isfinite(declared_duration) or not 0.0 < declared_duration <= FIXED_AUDIO_SECONDS:
        raise ValueError(
            f"scene {scene_id} duration must lie in (0,{FIXED_AUDIO_SECONDS}]"
        )
    duration_tolerance = 0.5 / asset.sample_rate + 1e-9
    if abs(declared_duration - asset.duration_seconds) > duration_tolerance:
        raise ValueError(
            f"scene {scene_id} duration mismatch: declared={declared_duration:.9f}, "
            f"decoded={asset.duration_seconds:.9f}"
        )
    if declared_frames != int(round(declared_duration * declared_rate)):
        raise ValueError(
            f"scene {scene_id} declared duration/rate/frame count are inconsistent"
        )


def _verify_events(
    row: Mapping[str, Any], asset: AudioAsset, *, scene_id: str
) -> list[dict[str, Any]]:
    events = list(row.get("events") or ())
    for index, event in enumerate(events):
        context = f"{scene_id}.events[{index}]"
        try:
            onset = float(event["onset_seconds"])
            offset = float(event["offset_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{context} has no valid onset/offset") from exc
        onset_frame = _grid_frame(onset, context=f"{context}.onset_seconds")
        offset_frame = _grid_frame(offset, context=f"{context}.offset_seconds")
        if offset_frame <= onset_frame:
            raise ValueError(f"{context} has a non-positive grid duration")
        sample_tolerance = 0.5 / asset.sample_rate + 1e-9
        if onset < -sample_tolerance or offset > asset.duration_seconds + sample_tolerance:
            raise ValueError(
                f"{context} lies outside decoded duration {asset.duration_seconds:.9f}"
            )
    return events


def _component_contract_present(events: Sequence[Mapping[str, Any]]) -> bool:
    fields = ("component_path", "component_sha256", "component_num_samples")
    return any(any(event.get(field) not in (None, "") for field in fields) for event in events)


def _verify_component_reconstruction(
    row: Mapping[str, Any],
    mixture: AudioAsset,
    *,
    scene_id: str,
    reader: OnceAudioReader,
    tolerance: float,
) -> float | None:
    events = list(row.get("events") or ())
    if not _component_contract_present(events):
        return None
    reconstruction = np.zeros(mixture.frames, dtype=np.float64)
    for index, event in enumerate(events):
        context = f"{scene_id}.events[{index}].component"
        required = ("component_path", "component_sha256", "component_num_samples")
        missing = [field for field in required if event.get(field) in (None, "")]
        if missing:
            raise ValueError(f"{context} has partial reconstruction metadata: missing {missing}")
        component = reader.read(
            event["component_path"],
            expected_sha256=event["component_sha256"],
            role="source_component",
            context=context,
            require_audible=True,
        )
        if component.sample_rate != mixture.sample_rate or component.channels != 1:
            raise ValueError(f"{context} geometry does not match mixture")
        if component.frames != int(event["component_num_samples"]):
            raise ValueError(
                f"{context} frame mismatch: declared={event['component_num_samples']} "
                f"decoded={component.frames}"
            )
        onset = float(event["onset_seconds"])
        expected_start = int(round(onset * mixture.sample_rate))
        if event.get("placement_start_sample") is None:
            raise ValueError(f"{context} has no placement_start_sample")
        start = int(event["placement_start_sample"])
        if start != expected_start:
            raise ValueError(
                f"{context} placement does not match onset: declared={start}, "
                f"expected={expected_start}"
            )
        expected_frames = int(
            round((float(event["offset_seconds"]) - onset) * mixture.sample_rate)
        )
        if component.frames != expected_frames:
            raise ValueError(
                f"{context} length does not match event interval: decoded={component.frames}, "
                f"expected={expected_frames}"
            )
        end = start + component.frames
        if start < 0 or end > mixture.frames:
            raise ValueError(f"{context} placement exceeds mixture bounds")
        reconstruction[start:end] += component.samples[:, 0].astype(np.float64)
    error = float(
        np.max(np.abs(reconstruction - mixture.samples[:, 0].astype(np.float64)))
    )
    if error > tolerance:
        raise ValueError(
            f"scene {scene_id} component reconstruction error {error:.9g} > {tolerance}"
        )
    return error


def _evidence_contracts(row: Mapping[str, Any]) -> list[tuple[str, str, str, str, str]]:
    contracts: list[tuple[str, str, str, str, str]] = []
    for name, evidence_prefix, residual_prefix in (
        ("evidence", "evidence", "residual"),
        ("oracle_evidence", "oracle_evidence", "oracle_residual"),
    ):
        fields = (
            f"{evidence_prefix}_path",
            f"{evidence_prefix}_sha256",
            f"{residual_prefix}_path",
            f"{residual_prefix}_sha256",
        )
        if any(row.get(field) not in (None, "") for field in fields):
            contracts.append((name, *fields))
    return contracts


def _verify_evidence_reconstruction(
    row: Mapping[str, Any],
    mixture: AudioAsset,
    *,
    scene_id: str,
    reader: OnceAudioReader,
    tolerance: float,
) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    for name, evidence_path, evidence_sha, residual_path, residual_sha in _evidence_contracts(row):
        missing = [
            field
            for field in (evidence_path, evidence_sha, residual_path, residual_sha)
            if row.get(field) in (None, "")
        ]
        if missing:
            raise ValueError(
                f"scene {scene_id} has partial {name} reconstruction metadata: missing {missing}"
            )
        evidence = reader.read(
            row[evidence_path],
            expected_sha256=row[evidence_sha],
            role=f"{name}_stem",
            context=f"scene {scene_id}.{name}",
            require_audible=True,
        )
        residual = reader.read(
            row[residual_path],
            expected_sha256=row[residual_sha],
            role=f"{name}_residual",
            context=f"scene {scene_id}.{name}_residual",
            require_audible=False,
        )
        for asset, role in ((evidence, name), (residual, f"{name}_residual")):
            if (
                asset.sample_rate != mixture.sample_rate
                or asset.frames != mixture.frames
                or asset.channels != mixture.channels
            ):
                raise ValueError(f"scene {scene_id} {role} geometry does not match mixture")
        reconstructed = evidence.samples.astype(np.float64) + residual.samples.astype(np.float64)
        error = float(
            np.max(np.abs(reconstructed - mixture.samples.astype(np.float64)))
        )
        if error > tolerance:
            raise ValueError(
                f"scene {scene_id} {name}+residual reconstruction error "
                f"{error:.9g} > {tolerance}"
            )
        results.append((name, error))
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--metadata-gate-receipt", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clipping-threshold", type=float, default=DEFAULT_CLIP_THRESHOLD)
    parser.add_argument("--minimum-rms", type=float, default=DEFAULT_MINIMUM_RMS)
    parser.add_argument(
        "--reconstruction-tolerance",
        type=float,
        default=DEFAULT_RECONSTRUCTION_TOLERANCE,
    )
    return parser.parse_args(argv)


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < float(args.clipping_threshold) <= DEFAULT_CLIP_THRESHOLD:
        raise ValueError(
            f"clipping-threshold must lie in (0,{DEFAULT_CLIP_THRESHOLD}]; "
            "the authorization gate may only be made stricter"
        )
    if not DEFAULT_MINIMUM_RMS <= float(args.minimum_rms) < float(args.clipping_threshold):
        raise ValueError(
            f"minimum-rms must lie in [{DEFAULT_MINIMUM_RMS}, clipping-threshold); "
            "the authorization gate may only be made stricter"
        )
    if not 0.0 < float(args.reconstruction_tolerance) <= DEFAULT_RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"reconstruction-tolerance must lie in "
            f"(0,{DEFAULT_RECONSTRUCTION_TOLERANCE}]; the authorization gate may only be made stricter"
        )

    manifests = {
        "train": args.train_manifest.resolve(strict=True),
        "dev": args.dev_manifest.resolve(strict=True),
        "test": args.test_manifest.resolve(strict=True),
    }
    gate_path = args.metadata_gate_receipt.resolve(strict=True)
    gate = _load_json(gate_path)
    gate_failures = _metadata_gate_failures(gate, manifests=manifests)
    base = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "debug_microfit_only": True,
            "paper_eligible": False,
            "gpu_used": False,
            "training_started": False,
        },
        "contract": {
            "train_scene_range": [TRAIN_SCENES_MIN, TRAIN_SCENES_MAX],
            "grid_frames": GRID_FRAMES,
            "frame_hop_seconds": FRAME_HOP_SECONDS,
            "maximum_audio_seconds": FIXED_AUDIO_SECONDS,
            "expected_sample_rate": EXPECTED_SAMPLE_RATE,
            "expected_channels": EXPECTED_CHANNELS,
            "clipping_threshold": float(args.clipping_threshold),
            "minimum_rms": float(args.minimum_rms),
            "reconstruction_tolerance": float(args.reconstruction_tolerance),
            "declared_frame_count_required": True,
            "declared_channel_count_required": True,
        },
        "inputs": {
            "metadata_gate": {"path": str(gate_path), "sha256": _sha256(gate_path)},
            "manifests": {
                split: {"path": str(path), "sha256": _sha256(path)}
                for split, path in manifests.items()
            },
            "audio_root": str(args.audio_root.resolve()),
        },
        "metadata_gate_validation": {
            "format": gate.get("format"),
            "passes": not gate_failures,
            "failures": gate_failures,
        },
    }
    if gate_failures:
        return {
            **base,
            "passes": False,
            "audio_training_authorized": False,
            "failures": gate_failures[:MAX_FAILURES_IN_RECEIPT],
            "failure_count": len(gate_failures),
            "scope": {**base["scope"], "audio_training_authorized": False},
            "audio_io": {
                "audio_files_opened": 0,
                "reason": "metadata gate validation failed before audio I/O",
            },
        }

    rows_by_split = {split: _read_jsonl(path) for split, path in manifests.items()}
    failures: list[str] = []
    train_count = len(rows_by_split["train"])
    if not TRAIN_SCENES_MIN <= train_count <= TRAIN_SCENES_MAX:
        failures.append(
            f"train scene count {train_count} is outside "
            f"[{TRAIN_SCENES_MIN},{TRAIN_SCENES_MAX}]"
        )
    gate_splits = gate.get("splits") or {}
    for split, rows in rows_by_split.items():
        gate_count = int((gate_splits.get(split) or {}).get("scenes", -1))
        if gate_count != len(rows):
            failures.append(
                f"metadata gate scene count mismatch for {split}: "
                f"gate={gate_count}, manifest={len(rows)}"
            )
    minimum_dev = int((gate.get("contract") or {}).get("minimum_dev_scenes", 1))
    minimum_test = int((gate.get("contract") or {}).get("minimum_test_scenes", 1))
    if len(rows_by_split["dev"]) < minimum_dev:
        failures.append(f"dev has fewer than metadata-gated minimum {minimum_dev}")
    if len(rows_by_split["test"]) < minimum_test:
        failures.append(f"test has fewer than metadata-gated minimum {minimum_test}")

    identity_failures, identity = _validate_manifest_identity(rows_by_split)
    failures.extend(identity_failures)
    if failures:
        return {
            **base,
            "passes": False,
            "audio_training_authorized": False,
            "failures": failures[:MAX_FAILURES_IN_RECEIPT],
            "failure_count": len(failures),
            "scope": {**base["scope"], "audio_training_authorized": False},
            "integrity": identity,
            "audio_io": {
                "audio_files_opened": 0,
                "reason": "manifest/identity validation failed before audio I/O",
            },
        }

    reader = OnceAudioReader(
        audio_root=args.audio_root,
        clipping_threshold=float(args.clipping_threshold),
        minimum_rms=float(args.minimum_rms),
    )
    path_failures, path_audit = _validate_resolved_audio_paths(
        rows_by_split, reader=reader
    )
    if path_failures:
        return {
            **base,
            "passes": False,
            "audio_training_authorized": False,
            "failures": path_failures[:MAX_FAILURES_IN_RECEIPT],
            "failure_count": len(path_failures),
            "scope": {**base["scope"], "audio_training_authorized": False},
            "integrity": identity,
            "resolved_paths": path_audit,
            "audio_io": {
                "audio_files_opened": 0,
                "reason": "exact path validation failed before audio I/O",
            },
        }
    split_stats: dict[str, dict[str, Any]] = {}
    component_scenes = 0
    component_errors: list[float] = []
    evidence_contract_counts: Counter[str] = Counter()
    evidence_errors: dict[str, list[float]] = defaultdict(list)
    total_events = 0
    for split, rows in rows_by_split.items():
        peaks: list[float] = []
        rms_values: list[float] = []
        frame_counts: list[int] = []
        durations: list[float] = []
        audited_scenes = 0
        split_events = 0
        for row_index, row in enumerate(rows):
            scene_id = str(row.get("scene_id") or f"{split}[{row_index}]")
            try:
                mixture = reader.read(
                    row.get("mixture_path"),
                    expected_sha256=row.get("audio_sha256"),
                    role=f"mixture_{split}",
                    context=f"scene {scene_id}.mixture",
                    require_audible=True,
                )
                _verify_mixture_declarations(row, mixture, scene_id=scene_id)
                events = _verify_events(row, mixture, scene_id=scene_id)
                split_events += len(events)
                total_events += len(events)
                component_error = _verify_component_reconstruction(
                    row,
                    mixture,
                    scene_id=scene_id,
                    reader=reader,
                    tolerance=float(args.reconstruction_tolerance),
                )
                if component_error is not None:
                    component_scenes += 1
                    component_errors.append(component_error)
                for name, error in _verify_evidence_reconstruction(
                    row,
                    mixture,
                    scene_id=scene_id,
                    reader=reader,
                    tolerance=float(args.reconstruction_tolerance),
                ):
                    evidence_contract_counts[name] += 1
                    evidence_errors[name].append(error)
                audited_scenes += 1
                peaks.append(mixture.peak)
                rms_values.append(mixture.rms)
                frame_counts.append(mixture.frames)
                durations.append(mixture.duration_seconds)
            except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
        split_stats[split] = {
            "declared_scenes": len(rows),
            "audited_scenes": audited_scenes,
            "events": split_events,
            "decoded_frames_min": min(frame_counts) if frame_counts else None,
            "decoded_frames_max": max(frame_counts) if frame_counts else None,
            "decoded_duration_seconds_min": min(durations) if durations else None,
            "decoded_duration_seconds_max": max(durations) if durations else None,
            "mixture_peak_max": max(peaks) if peaks else None,
            "mixture_rms_min": min(rms_values) if rms_values else None,
            "passes": audited_scenes == len(rows),
        }

    sorted_inventory = sorted(
        reader.inventory, key=lambda item: (item["path"], item["role"])
    )
    passes = not failures and all(summary["passes"] for summary in split_stats.values())
    return {
        **base,
        "passes": passes,
        "audio_training_authorized": passes,
        "failures": failures[:MAX_FAILURES_IN_RECEIPT],
        "failure_count": len(failures),
        "scope": {**base["scope"], "audio_training_authorized": passes},
        "integrity": identity,
        "resolved_paths": path_audit,
        "splits": split_stats,
        "temporal_grid": {
            "events_checked": total_events,
            "all_within_decoded_duration": passes,
            "all_on_fixed_grid": passes,
        },
        "reconstruction": {
            "source_components": {
                "scenes_with_contract": component_scenes,
                "scenes_without_contract": sum(len(rows) for rows in rows_by_split.values())
                - component_scenes,
                "maximum_absolute_error": max(component_errors) if component_errors else None,
                "tolerance": float(args.reconstruction_tolerance),
            },
            "evidence_residual": {
                name: {
                    "scenes_with_contract": count,
                    "maximum_absolute_error": max(evidence_errors[name]),
                    "tolerance": float(args.reconstruction_tolerance),
                }
                for name, count in sorted(evidence_contract_counts.items())
            },
        },
        "audio_io": {
            "audio_files_opened": reader.files_opened,
            "filesystem_opens_per_audio": 1,
            "roles": dict(sorted(reader.role_counts.items())),
            "inventory_entries": len(sorted_inventory),
            "inventory_sha256": _canonical_json_sha256(sorted_inventory),
            "path_reuse_allowed": False,
            "hash_and_decode_from_same_encoded_bytes": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    try:
        receipt = build_receipt(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "passes": False,
            "audio_training_authorized": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "failure_count": 1,
            "scope": {
                "debug_microfit_only": True,
                "paper_eligible": False,
                "gpu_used": False,
                "training_started": False,
                "audio_training_authorized": False,
            },
            "audio_io": {"audio_files_opened": 0},
        }
    _atomic_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    raise SystemExit(0 if main()["passes"] else 2)
