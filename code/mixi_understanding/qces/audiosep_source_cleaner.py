"""Build an auditable AudioSep-cleaned event source bank.

This is a sidecar data pipeline.  It deliberately does not reuse any QA
answer, QA score, or downstream model prediction.  An upstream source split is
copied verbatim and is never changed by the quality gate.  Frozen AudioSep and
its own frozen CLAP encoder are used only to clean and audit an event crop whose
official strong label is already known.

Durability is item-transactional: an accepted stem (and an optional audit
residual) is written atomically, then a content-bound fragment is committed.  Aggregate JSONL
manifests and shards are deterministic views of those fragments, so restarting
the command never repeats a valid separator call.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import scipy
from scipy import signal
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "qces_audiosep_clean_source_bank_v3"
FRAGMENT_FORMAT = "qces_audiosep_clean_source_fragment_v3"
QUALITY_PROTOCOL = "qces_audiosep_clean_quality_gate_v2_fixed_audible_support"
INPUT_FORMAT = "qces_audioset_strong_requested_crop_v1"
RESAMPLER_PROTOCOL = (
    "qces_scipy_resample_poly_float64_kaiser_beta5_"
    "constant_pad_round_half_to_even_length_v1"
)


class SourceCleanerError(RuntimeError):
    """Raised when the source-bank contract cannot be guaranteed."""


class CleanerBackend(Protocol):
    """Small dependency-injection boundary used by the official and test backends."""

    sample_rate: int

    def identity(self) -> Mapping[str, Any]: ...

    def separate(
        self, waveforms: Sequence[np.ndarray], prompts: Sequence[str]
    ) -> Sequence[np.ndarray]: ...

    def embed_audio(self, waveforms: Sequence[np.ndarray]) -> np.ndarray: ...

    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class QualityLevel:
    minimum_rms_dbfs: float
    minimum_peak_dbfs: float
    minimum_retained_energy_ratio: float
    maximum_retained_energy_ratio: float
    minimum_target_text_similarity: float
    minimum_target_residual_margin: float
    minimum_target_other_margin: float
    minimum_paraphrase_agreement: float
    maximum_clipped_sample_fraction: float


@dataclass(frozen=True)
class QualityGate:
    """Frozen thresholds; never tune these on QA or downstream accuracy.

    ``gold`` accepts any upstream ambiguity tier.  ``silver`` is deliberately
    available only to upstream tiers 0/1.  Thus AudioSep cannot make an
    annotation-ambiguous tier-2/3 crop look clean merely by suppressing a
    distractor.
    """

    gold: QualityLevel = QualityLevel(
        minimum_rms_dbfs=-45.0,
        minimum_peak_dbfs=-35.0,
        minimum_retained_energy_ratio=0.005,
        maximum_retained_energy_ratio=2.50,
        minimum_target_text_similarity=0.15,
        minimum_target_residual_margin=0.02,
        minimum_target_other_margin=0.00,
        minimum_paraphrase_agreement=0.75,
        maximum_clipped_sample_fraction=0.001,
    )
    silver: QualityLevel = QualityLevel(
        minimum_rms_dbfs=-55.0,
        minimum_peak_dbfs=-45.0,
        minimum_retained_energy_ratio=0.001,
        maximum_retained_energy_ratio=4.00,
        minimum_target_text_similarity=0.05,
        minimum_target_residual_margin=-0.05,
        minimum_target_other_margin=-0.08,
        minimum_paraphrase_agreement=0.55,
        maximum_clipped_sample_fraction=0.005,
    )
    protocol: str = QUALITY_PROTOCOL


@dataclass(frozen=True)
class SourceBankConfig:
    manifest_paths: tuple[Path, ...]
    output_dir: Path
    batch_size: int = 4
    shard_size: int = 256
    max_items: int = 0
    target_train_per_class: int = 100
    target_eval_per_class: int = 20
    # The residual is used in-memory by every quality gate, but the downstream
    # scene builder only consumes the clean stem.  Keeping residual artifacts
    # opt-in avoids roughly doubling a 24,000-crop bank on a constrained disk.
    store_residual: bool = False
    quality_gate: QualityGate = QualityGate()

    def __post_init__(self) -> None:
        if not self.manifest_paths:
            raise ValueError("at least one requested-crop manifest is required")
        if self.batch_size < 1 or self.shard_size < 1:
            raise ValueError("batch_size and shard_size must be positive")
        if self.max_items < 0:
            raise ValueError("max_items must be non-negative")
        if self.target_train_per_class < 1 or self.target_eval_per_class < 1:
            raise ValueError("class quotas must be positive")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, block_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _waveform_f32le_sha256(waveform: np.ndarray) -> str:
    """Hash decoded mono samples with an explicit, portable byte convention."""

    value = np.ascontiguousarray(np.asarray(waveform, dtype="<f4"))
    return sha256_bytes(value.tobytes(order="C"))


def resampler_identity(*, target_sample_rate: int) -> dict[str, Any]:
    """Content-bound identity for the fixed AudioSep input normalizer."""

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")
    return {
        "protocol": RESAMPLER_PROTOCOL,
        "implementation": "scipy.signal.resample_poly",
        "scipy_version": str(scipy.__version__),
        "numpy_version": str(np.__version__),
        "target_sample_rate": int(target_sample_rate),
        "working_dtype": "float64",
        "output_dtype": "float32_little_endian_hash_convention",
        "window": {"name": "kaiser", "beta": 5.0},
        "padtype": "constant",
        "cval": 0.0,
        "output_length": "round_half_to_even(input_samples*target_rate/source_rate)",
    }


def deterministic_resample_mono(
    waveform: np.ndarray,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize a decoded mono crop to the frozen separator sample rate.

    AudioSet mirrors contain both 44.1 and 48 kHz FLACs while AudioSep is a
    32 kHz model.  Resampling is therefore part of the content-bound cleaning
    transaction, not an implicit backend side effect.  The rational factors,
    fixed window/padding and exact output-length rule make the operation
    replayable and keep timestamps expressed in seconds unchanged.
    """

    value = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
    if value.ndim != 1 or value.size <= 0 or not np.isfinite(value).all():
        raise SourceCleanerError("resampler requires a finite non-empty mono waveform")
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise SourceCleanerError("resampler sample rates must be positive")
    divisor = math.gcd(int(source_sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // divisor
    down = int(source_sample_rate) // divisor
    expected_samples = max(
        1,
        int(round(value.size * float(target_sample_rate) / float(source_sample_rate))),
    )
    if source_sample_rate == target_sample_rate:
        output = value.copy()
    else:
        output64 = signal.resample_poly(
            np.asarray(value, dtype=np.float64),
            up,
            down,
            window=("kaiser", 5.0),
            padtype="constant",
            cval=0.0,
        )
        if output64.size > expected_samples:
            output64 = output64[:expected_samples]
        elif output64.size < expected_samples:
            output64 = np.pad(
                output64,
                (0, expected_samples - output64.size),
                mode="constant",
                constant_values=0.0,
            )
        output = np.ascontiguousarray(output64, dtype=np.float32)
    if output.size != expected_samples or not np.isfinite(output).all():
        raise SourceCleanerError("deterministic resampler produced invalid output")
    source_duration = value.size / float(source_sample_rate)
    output_duration = output.size / float(target_sample_rate)
    return output, {
        **resampler_identity(target_sample_rate=target_sample_rate),
        "source_sample_rate": int(source_sample_rate),
        "source_num_samples": int(value.size),
        "source_decoded_mono_f32le_sha256": _waveform_f32le_sha256(value),
        "up_factor": up,
        "down_factor": down,
        "resampling_applied": bool(source_sample_rate != target_sample_rate),
        "separator_input_num_samples": int(output.size),
        "separator_input_f32le_sha256": _waveform_f32le_sha256(output),
        "source_duration_seconds": source_duration,
        "separator_input_duration_seconds": output_duration,
        "duration_error_seconds": output_duration - source_duration,
    }


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
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
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_bytes(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_bytes(
        path,
        b"".join(_canonical_json_bytes(row) + b"\n" for row in rows),
    )


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_audio_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    for base in (PROJECT_ROOT, manifest_path.parent, Path.cwd()):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return (manifest_path.parent / candidate).resolve()


def _safe_component(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return normalized[:96] or fallback


def _split_identity(row: Mapping[str, Any]) -> str:
    split = str(row.get("split_lock") or "")
    if not split:
        # Legacy requested-crop manifests may only carry metadata_split.  Keep
        # it explicit rather than silently inventing a train/dev allocation.
        split = str(row.get("metadata_split") or "")
    if split not in {
        "train",
        "dev",
        "test",
        "eval",
        "unassigned_train_pool",
    }:
        raise SourceCleanerError(f"unsupported or empty source split: {split!r}")
    return split


def _canonical_display_name(row: Mapping[str, Any]) -> str:
    label = str(row.get("coverage_label") or "")
    mid = str(row.get("coverage_mid") or "")
    candidates: list[Mapping[str, Any]] = []
    events = row.get("events")
    if isinstance(events, list):
        candidates.extend(value for value in events if isinstance(value, Mapping))
    request = row.get("crop_request")
    if isinstance(request, Mapping):
        for key in ("strong_annotations", "selected_ontology_annotations"):
            values = request.get(key)
            if isinstance(values, list):
                candidates.extend(
                    value for value in values if isinstance(value, Mapping)
                )
    for event in candidates:
        event_label = str(event.get("label") or "")
        event_mid = str(event.get("mid") or event.get("audioset_mid") or "")
        if event_label == label and (not mid or not event_mid or event_mid == mid):
            display = str(event.get("display_name") or "").strip()
            if display:
                return display
    # The joint plan uses normalized AudioSet labels, so this is still a
    # deterministic canonical rendering if an old crop omitted display_name.
    if not label:
        raise SourceCleanerError("requested crop has no coverage_label")
    return label.replace("_", " ")


def canonical_prompts(row: Mapping[str, Any]) -> tuple[str, str]:
    display = _canonical_display_name(row)
    return display, f"the sound of {display}"


def _other_display_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    target = str(row.get("coverage_label") or "")
    names: dict[str, str] = {}
    events = row.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            label = str(event.get("label") or "")
            if not label or label == target:
                continue
            display = str(event.get("display_name") or label.replace("_", " "))
            names[label] = display
    return tuple(names[label] for label in sorted(names))


def load_requested_crops(
    paths: Sequence[Path], *, max_items: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    source_splits: dict[str, str] = {}
    item_ids: set[str] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise SourceCleanerError(f"requested-crop manifest does not exist: {path}")
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    source = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SourceCleanerError(
                        f"invalid JSON at {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(source, dict):
                    raise SourceCleanerError(
                        f"expected object at {path}:{line_number}"
                    )
                if str(source.get("format") or "") != INPUT_FORMAT:
                    raise SourceCleanerError(
                        f"unexpected format at {path}:{line_number}: "
                        f"{source.get('format')!r}"
                    )
                source_video_id = str(
                    source.get("source_video_id") or source.get("video_id") or ""
                )
                item_id = str(
                    source.get("materialization_item_id")
                    or (source.get("crop_request") or {}).get("selection_key")
                    or ""
                )
                if not source_video_id or not item_id:
                    raise SourceCleanerError(
                        f"missing source_video_id/item id at {path}:{line_number}"
                    )
                if item_id in item_ids:
                    raise SourceCleanerError(f"duplicate crop item id: {item_id}")
                item_ids.add(item_id)
                split = _split_identity(source)
                previous = source_splits.setdefault(source_video_id, split)
                if previous != split:
                    raise SourceCleanerError(
                        "source_video_id occurs in multiple splits: "
                        f"{source_video_id} -> {previous}, {split}"
                    )
                tier = int(source.get("ambiguity_tier", -1))
                if tier not in {0, 1, 2, 3}:
                    raise SourceCleanerError(
                        f"invalid ambiguity_tier={tier} for {item_id}"
                    )
                audio_path = resolve_audio_path(path, str(source.get("mixture_path") or ""))
                if not audio_path.is_file():
                    raise SourceCleanerError(f"crop audio does not exist: {audio_path}")
                actual_hash = sha256_file(audio_path)
                declared_hash = str(source.get("audio_sha256") or "")
                if declared_hash and declared_hash != actual_hash:
                    raise SourceCleanerError(
                        f"crop audio hash mismatch for {item_id}: "
                        f"{actual_hash} != {declared_hash}"
                    )
                normalized = dict(source)
                normalized["_input_manifest"] = str(path)
                normalized["_input_manifest_sha256"] = sha256_file(path)
                normalized["_audio_path"] = str(audio_path)
                normalized["_audio_sha256"] = actual_hash
                normalized["_source_split"] = split
                normalized["_source_video_id"] = source_video_id
                normalized["_item_id"] = item_id
                normalized["_canonical_display_name"] = _canonical_display_name(source)
                rows.append(normalized)
                count += 1
        receipts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": count,
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["ambiguity_tier"]),
            str(row["_source_split"]),
            str(row["coverage_label"]),
            str(row["_source_video_id"]),
            str(row["_item_id"]),
        )
    )
    if max_items:
        rows = rows[:max_items]
    input_hash = sha256_bytes(
        _canonical_json_bytes(
            [
                {
                    "item_id": row["_item_id"],
                    "row": {
                        key: value
                        for key, value in row.items()
                        if not key.startswith("_")
                    },
                    "audio_sha256": row["_audio_sha256"],
                }
                for row in rows
            ]
        )
    )
    return rows, receipts, input_hash


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise SourceCleanerError("embedding has zero/invalid norm")
    return value / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_unit(left), _unit(right)))


def _waveform_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine for an informational waveform metric, including silent outputs."""

    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left_value))
    right_norm = float(np.linalg.norm(right_value))
    if left_norm <= 1e-12 and right_norm <= 1e-12:
        return 1.0
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left_value, right_value) / (left_norm * right_norm))


def _rms_dbfs(waveform: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    return float(20.0 * math.log10(max(rms, 1e-12)))


def _peak_dbfs(waveform: np.ndarray) -> float:
    peak = float(np.max(np.abs(waveform), initial=0.0))
    return float(20.0 * math.log10(max(peak, 1e-12)))


def _quality_failures(
    metrics: Mapping[str, Any], level: QualityLevel
) -> list[str]:
    failures: list[str] = []
    checks = (
        ("stem_rms_dbfs", float(metrics["stem_rms_dbfs"]) >= level.minimum_rms_dbfs),
        ("stem_peak_dbfs", float(metrics["stem_peak_dbfs"]) >= level.minimum_peak_dbfs),
        (
            "retained_energy_ratio_min",
            float(metrics["retained_energy_ratio"])
            >= level.minimum_retained_energy_ratio,
        ),
        (
            "retained_energy_ratio_max",
            float(metrics["retained_energy_ratio"])
            <= level.maximum_retained_energy_ratio,
        ),
        (
            "target_text_similarity",
            float(metrics["target_text_similarity"])
            >= level.minimum_target_text_similarity,
        ),
        (
            "target_residual_margin",
            float(metrics["target_residual_margin"])
            >= level.minimum_target_residual_margin,
        ),
        (
            "paraphrase_agreement",
            float(metrics["paraphrase_agreement"])
            >= level.minimum_paraphrase_agreement,
        ),
        (
            "clipped_sample_fraction",
            float(metrics["maximum_clipped_sample_fraction"])
            <= level.maximum_clipped_sample_fraction,
        ),
    )
    for name, passed in checks:
        if not passed:
            failures.append(name)
    other_margin = metrics.get("target_other_margin")
    if other_margin is not None and float(other_margin) < level.minimum_target_other_margin:
        failures.append("target_other_margin")
    return failures


def apply_quality_gate(
    metrics: Mapping[str, Any], *, ambiguity_tier: int, gate: QualityGate
) -> dict[str, Any]:
    gold_failures = _quality_failures(metrics, gate.gold)
    silver_failures = _quality_failures(metrics, gate.silver)
    if not gold_failures:
        acceptance = "gold"
    elif ambiguity_tier <= 1 and not silver_failures:
        acceptance = "silver"
    else:
        acceptance = "rejected"
    return {
        "accepted": acceptance != "rejected",
        "acceptance_tier": acceptance,
        "quality_protocol": gate.protocol,
        "gold_failures": gold_failures,
        "silver_failures": silver_failures,
        "silver_disallowed_by_upstream_ambiguity": (
            ambiguity_tier > 1 and not silver_failures and bool(gold_failures)
        ),
    }


def quality_metrics(
    *,
    mixture: np.ndarray,
    stem: np.ndarray,
    alternate_stem: np.ndarray,
    residual: np.ndarray,
    target_text_vector: np.ndarray,
    other_text_vectors: Sequence[np.ndarray],
    stem_audio_vector: np.ndarray,
    alternate_audio_vector: np.ndarray,
    residual_audio_vector: np.ndarray,
) -> dict[str, Any]:
    mixture = np.asarray(mixture, dtype=np.float32)
    stem = np.asarray(stem, dtype=np.float32)
    alternate_stem = np.asarray(alternate_stem, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    if not (
        mixture.ndim == stem.ndim == alternate_stem.ndim == residual.ndim == 1
        and mixture.shape == stem.shape == alternate_stem.shape == residual.shape
    ):
        raise SourceCleanerError("mixture and separator outputs must be equal 1-D arrays")
    if not all(
        np.isfinite(value).all()
        for value in (mixture, stem, alternate_stem, residual)
    ):
        raise SourceCleanerError("audio contains NaN or Inf")
    mix_energy = float(np.sum(np.square(mixture, dtype=np.float64)))
    stem_energy = float(np.sum(np.square(stem, dtype=np.float64)))
    target_similarity = _cosine(stem_audio_vector, target_text_vector)
    residual_similarity = _cosine(residual_audio_vector, target_text_vector)
    other_similarities = [
        _cosine(stem_audio_vector, value) for value in other_text_vectors
    ]
    stem_clip = float(np.mean(np.abs(stem) >= 1.0))
    residual_clip = float(np.mean(np.abs(residual) >= 1.0))
    return {
        "stem_rms_dbfs": _rms_dbfs(stem),
        "stem_peak_dbfs": _peak_dbfs(stem),
        "mixture_rms_dbfs": _rms_dbfs(mixture),
        "retained_energy_ratio": stem_energy / max(mix_energy, 1e-12),
        "target_text_similarity": target_similarity,
        "residual_target_text_similarity": residual_similarity,
        "target_residual_margin": target_similarity - residual_similarity,
        "maximum_other_text_similarity": (
            max(other_similarities) if other_similarities else None
        ),
        "target_other_margin": (
            target_similarity - max(other_similarities)
            if other_similarities
            else None
        ),
        "paraphrase_agreement": _cosine(
            stem_audio_vector, alternate_audio_vector
        ),
        "primary_alternate_waveform_cosine": _waveform_cosine(
            stem, alternate_stem
        ),
        "mixture_consistency_rmse": float(
            np.sqrt(np.mean(np.square(mixture - stem - residual, dtype=np.float64)))
        ),
        "stem_clipped_sample_fraction": stem_clip,
        "residual_clipped_sample_fraction": residual_clip,
        "maximum_clipped_sample_fraction": max(stem_clip, residual_clip),
    }


def _encode_flac(waveform: np.ndarray, sample_rate: int) -> bytes:
    import io

    buffer = io.BytesIO()
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0 - 2 ** -23)
    sf.write(buffer, clipped, sample_rate, format="FLAC", subtype="PCM_24")
    return buffer.getvalue()


def _validate_artifact(path: Path, expected_hash: str) -> bool:
    if not path.is_file() or sha256_file(path) != expected_hash:
        return False
    try:
        info = sf.info(path)
    except Exception:
        return False
    return int(info.channels) == 1 and int(info.frames) > 0


def _transaction_hash(
    row: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
    input_normalizer_identity: Mapping[str, Any],
    gate: QualityGate,
    *,
    store_residual: bool,
) -> str:
    public_row = {key: value for key, value in row.items() if not key.startswith("_")}
    return sha256_bytes(
        _canonical_json_bytes(
            {
                "format": FORMAT,
                "row": public_row,
                "source_audio_sha256": row["_audio_sha256"],
                "backend": backend_identity,
                "input_normalizer": input_normalizer_identity,
                "gate": asdict(gate),
                "store_residual": bool(store_residual),
            }
        )
    )


def _load_resume_fragment(
    path: Path, *, transaction_hash: str, output_dir: Path
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(record, dict)
        or record.get("format") != FRAGMENT_FORMAT
        or record.get("transaction_sha256") != transaction_hash
    ):
        return None
    if bool(record.get("accepted")):
        for path_key, hash_key in (
            ("stem_path", "stem_sha256"),
            ("residual_path", "residual_sha256"),
        ):
            raw_value = str(record.get(path_key) or "")
            # Residual persistence is intentionally optional; an empty pair
            # is not a corrupt transaction.
            if not raw_value and not str(record.get(hash_key) or ""):
                continue
            value = Path(raw_value)
            artifact = value if value.is_absolute() else (PROJECT_ROOT / value)
            if not artifact.is_file():
                artifact = output_dir / value
            if not _validate_artifact(artifact.resolve(), str(record.get(hash_key) or "")):
                return None
    return record


def _coverage_interval(
    row: Mapping[str, Any], *, duration_seconds: float
) -> tuple[float, float]:
    """Return the exact coverage-event interval on the materialized crop grid.

    The crop manifest keeps every intersecting strong annotation, so matching
    only by label can select the wrong occurrence when a class repeats.  The
    requested source interval is therefore used as an identity tie-breaker
    whenever it is present.  Test/legacy manifests without source coordinates
    still have an unambiguous label+MID fallback.
    """

    label = str(row.get("coverage_label") or "")
    mid = str(row.get("coverage_mid") or "")
    request = row.get("crop_request")
    request = request if isinstance(request, Mapping) else {}
    requested_onset = request.get("event_onset_seconds")
    requested_offset = request.get("event_offset_seconds")
    candidates: list[tuple[float, float, float]] = []
    for raw in row.get("events") or []:
        if not isinstance(raw, Mapping) or str(raw.get("label") or "") != label:
            continue
        event_mid = str(raw.get("audioset_mid") or raw.get("mid") or "")
        if mid and event_mid and event_mid != mid:
            continue
        onset = float(raw.get("onset_seconds", -1.0))
        offset = float(raw.get("offset_seconds", -1.0))
        if onset < 0.0 or offset <= onset:
            continue
        identity_distance = 0.0
        if requested_onset is not None and requested_offset is not None:
            source_onset = float(raw.get("source_onset_seconds", math.inf))
            source_offset = float(raw.get("source_offset_seconds", math.inf))
            identity_distance = abs(source_onset - float(requested_onset)) + abs(
                source_offset - float(requested_offset)
            )
        candidates.append((identity_distance, onset, offset))
    if not candidates:
        raise SourceCleanerError(
            f"coverage event {label!r}/{mid!r} is absent from {row.get('_item_id')!r}"
        )
    _, onset, offset = min(candidates, key=lambda value: value[0])
    onset = max(0.0, min(float(duration_seconds), onset))
    offset = max(0.0, min(float(duration_seconds), offset))
    if offset <= onset:
        raise SourceCleanerError(
            f"coverage interval is empty after crop clipping for {row.get('_item_id')!r}"
        )
    return onset, offset


def estimate_audible_interval(
    waveform: np.ndarray,
    sample_rate: int,
    annotation_interval: tuple[float, float],
    *,
    window_seconds: float = 0.025,
    hop_seconds: float = 0.010,
    relative_floor_db: float = -30.0,
    absolute_floor_dbfs: float = -55.0,
    padding_seconds: float = 0.020,
) -> tuple[float, float]:
    """Trim annotation-edge silence using the *accepted stem* energy support.

    The returned interval never expands outside the strong annotation.  This
    matters because the synthetic scene builder treats its source interval as
    exact temporal ground truth; copying a coarse crop boundary would teach
    the detector that silent leading/trailing frames are active evidence.
    """

    value = np.asarray(waveform, dtype=np.float32)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise SourceCleanerError("audible interval requires a finite mono waveform")
    if sample_rate <= 0:
        raise SourceCleanerError("audible interval requires a positive sample rate")
    onset, offset = annotation_interval
    start = max(0, min(value.size - 1, int(math.floor(onset * sample_rate))))
    end = max(start + 1, min(value.size, int(math.ceil(offset * sample_rate))))
    window = max(1, int(round(window_seconds * sample_rate)))
    hop = max(1, int(round(hop_seconds * sample_rate)))
    starts = list(range(start, max(start + 1, end - window + 1), hop))
    final_start = max(start, end - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    rms = np.asarray(
        [
            math.sqrt(
                float(
                    np.mean(
                        np.square(value[index : min(end, index + window)], dtype=np.float64)
                    )
                )
            )
            for index in starts
        ],
        dtype=np.float64,
    )
    peak_rms = float(rms.max(initial=0.0))
    threshold = max(
        10.0 ** (absolute_floor_dbfs / 20.0),
        peak_rms * 10.0 ** (relative_floor_db / 20.0),
    )
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        # The upstream quality gate will normally reject such a stem.  The
        # annotation fallback keeps the helper total and auditable instead of
        # inventing a zero-length event.
        return float(onset), float(offset)
    support_start = starts[int(active[0])] / sample_rate - padding_seconds
    support_end = (
        min(end, starts[int(active[-1])] + window) / sample_rate + padding_seconds
    )
    support_start = max(float(onset), support_start)
    support_end = min(float(offset), support_end)
    if support_end <= support_start:
        return float(onset), float(offset)
    return float(support_start), float(support_end)


def _prepare_record(
    *,
    row: Mapping[str, Any],
    transaction_hash: str,
    backend_identity: Mapping[str, Any],
    input_audio_audit: Mapping[str, Any],
    sample_rate: int,
    mixture: np.ndarray,
    stem: np.ndarray,
    alternate_stem: np.ndarray,
    residual: np.ndarray,
    metrics: Mapping[str, Any],
    gate_result: Mapping[str, Any],
    output_dir: Path,
    store_residual: bool,
) -> dict[str, Any]:
    primary_prompt, paraphrase = canonical_prompts(row)
    accepted = bool(gate_result["accepted"])
    item_hash = sha256_bytes(
        f"{row['_source_video_id']}:{row['_item_id']}:{row['coverage_label']}".encode(
            "utf-8"
        )
    )
    stem_path: Path | None = None
    residual_path: Path | None = None
    stem_sha = ""
    residual_sha = ""
    encoded_consistency: float | None = None
    if accepted:
        split_component = _safe_component(str(row["_source_split"]), fallback="split")
        label_component = _safe_component(str(row["coverage_label"]), fallback="label")
        audio_dir = output_dir / "audio" / split_component / label_component
        stem_path = audio_dir / f"{item_hash[:32]}.stem.flac"
        residual_path = (
            audio_dir / f"{item_hash[:32]}.residual.flac"
            if store_residual
            else None
        )
        stem_payload = _encode_flac(stem, sample_rate)
        residual_payload = _encode_flac(residual, sample_rate)
        stem_sha = sha256_bytes(stem_payload)
        residual_sha = sha256_bytes(residual_payload) if store_residual else ""
        if not _validate_artifact(stem_path, stem_sha):
            atomic_bytes(stem_path, stem_payload)
        if residual_path is not None and not _validate_artifact(
            residual_path, residual_sha
        ):
            atomic_bytes(residual_path, residual_payload)
        decoded_stem, _ = sf.read(stem_path, dtype="float32", always_2d=False)
        if residual_path is not None:
            decoded_residual, _ = sf.read(
                residual_path, dtype="float32", always_2d=False
            )
        else:
            # Measure the same PCM24 round-trip consistency without retaining
            # an artifact that no downstream stage consumes.
            import io

            decoded_residual, _ = sf.read(
                io.BytesIO(residual_payload), dtype="float32", always_2d=False
            )
        encoded_consistency = float(
            np.sqrt(
                np.mean(
                    np.square(
                        mixture - decoded_stem - decoded_residual,
                        dtype=np.float64,
                    )
                )
            )
        )
    annotations = [
        dict(value)
        for value in row.get("events", [])
        if isinstance(value, Mapping)
    ]
    duration_seconds = len(mixture) / sample_rate
    annotation_onset, annotation_offset = _coverage_interval(
        row, duration_seconds=duration_seconds
    )
    active_onset, active_offset = estimate_audible_interval(
        stem,
        sample_rate,
        (annotation_onset, annotation_offset),
    )
    portable_stem = _portable_path(stem_path) if stem_path else ""
    acceptance_tier = str(gate_result["acceptance_tier"])
    return {
        "format": FRAGMENT_FORMAT,
        "source_bank_format": FORMAT,
        "transaction_sha256": transaction_hash,
        "item_id": str(row["_item_id"]),
        # Canonical single-event source-bank fields consumed directly by the
        # clean scene builder.  Keeping these beside the richer audit fields
        # prevents a lossy, hand-written conversion step between pipelines.
        "source_id": str(row["_item_id"]),
        "label": str(row["coverage_label"]),
        "source_video_id": str(row["_source_video_id"]),
        "video_id": str(row.get("video_id") or row["_source_video_id"]),
        "scene_id": str(row.get("scene_id") or ""),
        "source_split": str(row["_source_split"]),
        "split_lock": str(row.get("split_lock") or ""),
        "metadata_split": str(row.get("metadata_split") or ""),
        "hf_split": str(row.get("hf_split") or ""),
        "materialization_source_route": str(row.get("source_route") or ""),
        "materialization_hf_dataset": str(row.get("hf_dataset") or ""),
        "materialization_hf_revision": str(row.get("hf_revision") or ""),
        "coverage_label": str(row["coverage_label"]),
        "coverage_mid": str(row.get("coverage_mid") or ""),
        "canonical_display_name": str(row["_canonical_display_name"]),
        "canonical_prompt": primary_prompt,
        "canonical_paraphrase": paraphrase,
        "ambiguity_tier": int(row["ambiguity_tier"]),
        "ambiguity_tier_name": str(row.get("ambiguity_tier_name") or ""),
        "all_intersecting_strong_annotations": annotations,
        "source_crop_start_seconds": float(
            row.get("source_crop_start_seconds", 0.0)
        ),
        "source_crop_end_seconds": float(
            row.get("source_crop_end_seconds", row.get("duration_seconds", 0.0))
        ),
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "input_sample_rate": int(input_audio_audit["source_sample_rate"]),
        "input_num_samples": int(input_audio_audit["source_num_samples"]),
        "input_decoded_mono_f32le_sha256": str(
            input_audio_audit["source_decoded_mono_f32le_sha256"]
        ),
        "resampling_applied": bool(input_audio_audit["resampling_applied"]),
        "separator_input_sample_rate": int(
            input_audio_audit["target_sample_rate"]
        ),
        "separator_input_num_samples": int(
            input_audio_audit["separator_input_num_samples"]
        ),
        "separator_input_f32le_sha256": str(
            input_audio_audit["separator_input_f32le_sha256"]
        ),
        "separator_input_duration_seconds": float(
            input_audio_audit["separator_input_duration_seconds"]
        ),
        "resample_duration_error_seconds": float(
            input_audio_audit["duration_error_seconds"]
        ),
        "input_normalizer": dict(input_audio_audit),
        "source_audio_path": str(row.get("mixture_path") or ""),
        "source_audio_sha256": str(row["_audio_sha256"]),
        "stem_path": portable_stem,
        "stem_sha256": stem_sha,
        "audio_path": portable_stem,
        "source_path": portable_stem,
        "source_sha256": stem_sha,
        "active_onset_seconds": active_onset,
        "active_offset_seconds": active_offset,
        "strong_annotation_onset_seconds": annotation_onset,
        "strong_annotation_offset_seconds": annotation_offset,
        "audible_support_fraction_of_annotation": (
            (active_offset - active_onset)
            / max(annotation_offset - annotation_onset, 1e-12)
        ),
        "audible_support_method": (
            "accepted_audiosep_stem_rms_25ms_hop10ms_"
            "max_relative_minus30db_absolute_minus55db_pad20ms"
        ),
        "cleanliness_passed": accepted,
        "audibility_passed": accepted,
        "cleanliness_tier": acceptance_tier,
        "audibility_score": float(metrics["target_text_similarity"]),
        "residual_path": _portable_path(residual_path) if residual_path else "",
        "residual_sha256": residual_sha,
        "encoded_mixture_consistency_rmse": encoded_consistency,
        "accepted": accepted,
        "acceptance_tier": acceptance_tier,
        "quality_gate": dict(gate_result),
        "quality_metrics": dict(metrics),
        "provenance": {
            "input_manifest": str(row["_input_manifest"]),
            "input_manifest_sha256": str(row["_input_manifest_sha256"]),
            "source_audio_sha256": str(row["_audio_sha256"]),
            "materialization_source_route": str(row.get("source_route") or ""),
            "materialization_hf_dataset": str(row.get("hf_dataset") or ""),
            "materialization_hf_revision": str(row.get("hf_revision") or ""),
            "audiosep_backend": dict(backend_identity),
            "input_normalizer": dict(input_audio_audit),
            "quality_protocol": QUALITY_PROTOCOL,
            "selection_uses_qa_answer": False,
            "selection_uses_downstream_accuracy": False,
            "split_copied_before_quality_gate": True,
            "residual_definition": "source_crop_minus_primary_audiosep_stem",
            "residual_artifact_stored": bool(store_residual),
        },
    }


def _write_shards(
    directory: Path, rows: Sequence[Mapping[str, Any]], *, shard_size: int
) -> list[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    expected: set[Path] = set()
    for index, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        path = directory / f"shard-{index:06d}.jsonl"
        atomic_jsonl(path, shard_rows)
        expected.add(path.resolve())
        receipts.append(
            {
                "path": _portable_path(path),
                "sha256": sha256_file(path),
                "rows": len(shard_rows),
            }
        )
    # Stale files from an interrupted previous view must not be consumed by a
    # glob.  Only this pipeline's exact shard prefix inside its own directory is
    # eligible for removal.
    for path in directory.glob("shard-*.jsonl"):
        if path.resolve() not in expected:
            path.unlink()
    return receipts


def _quota_report(
    rows: Sequence[Mapping[str, Any]], *, train_target: int, eval_target: int
) -> dict[str, Any]:
    labels = sorted({str(row["coverage_label"]) for row in rows})
    input_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    accepted_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    accepted_tiers: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        metadata_split = str(row.get("metadata_split") or "")
        label = str(row["coverage_label"])
        source = str(row["source_video_id"])
        input_sources[(metadata_split, label)].add(source)
        if bool(row["accepted"]):
            accepted_sources[(metadata_split, label)].add(source)
            accepted_tiers[
                (metadata_split, label, str(row["acceptance_tier"]))
            ] += 1
    per_class: list[dict[str, Any]] = []
    for label in labels:
        train_count = len(accepted_sources[("train", label)])
        eval_count = len(accepted_sources[("eval", label)])
        per_class.append(
            {
                "label": label,
                "input_train_unique_videos": len(input_sources[("train", label)]),
                "input_eval_unique_videos": len(input_sources[("eval", label)]),
                "accepted_train_unique_videos": train_count,
                "accepted_eval_unique_videos": eval_count,
                "gold_train_items": accepted_tiers[("train", label, "gold")],
                "silver_train_items": accepted_tiers[("train", label, "silver")],
                "gold_eval_items": accepted_tiers[("eval", label, "gold")],
                "silver_eval_items": accepted_tiers[("eval", label, "silver")],
                "target_train_unique_videos": train_target,
                "target_eval_unique_videos": eval_target,
                "missing_train_unique_videos": max(0, train_target - train_count),
                "missing_eval_unique_videos": max(0, eval_target - eval_count),
                "quota_ready": train_count >= train_target and eval_count >= eval_target,
            }
        )
    return {
        "class_count": len(labels),
        "classes_quota_ready": sum(bool(row["quota_ready"]) for row in per_class),
        "target_train_unique_videos_per_class": train_target,
        "target_eval_unique_videos_per_class": eval_target,
        "per_class": per_class,
    }


def build_source_bank(
    config: SourceBankConfig, backend: CleanerBackend
) -> dict[str, Any]:
    """Run or resume source cleaning and return the final receipt."""

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, input_receipts, input_hash = load_requested_crops(
        config.manifest_paths, max_items=config.max_items
    )
    backend_identity = dict(backend.identity())
    if not backend_identity:
        raise SourceCleanerError("backend identity must not be empty")
    if int(getattr(backend, "sample_rate", 0)) <= 0:
        raise SourceCleanerError("backend sample_rate must be positive")
    input_normalizer_identity = resampler_identity(
        target_sample_rate=int(backend.sample_rate)
    )
    transactions = {
        str(row["_item_id"]): _transaction_hash(
            row,
            backend_identity,
            input_normalizer_identity,
            config.quality_gate,
            store_residual=config.store_residual,
        )
        for row in rows
    }
    fragments: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for row in rows:
        item_id = str(row["_item_id"])
        fragment_path = output_dir / "fragments" / f"{transactions[item_id]}.json"
        resumed = _load_resume_fragment(
            fragment_path,
            transaction_hash=transactions[item_id],
            output_dir=output_dir,
        )
        if resumed is None:
            pending.append(row)
        else:
            fragments[item_id] = resumed

    processed_now = 0
    separator_calls_now = 0
    for start in range(0, len(pending), config.batch_size):
        batch = pending[start : start + config.batch_size]
        mixtures: list[np.ndarray] = []
        input_audio_audits: list[dict[str, Any]] = []
        for row in batch:
            waveform, source_sample_rate = sf.read(
                Path(str(row["_audio_path"])), dtype="float32", always_2d=True
            )
            mono = np.asarray(waveform.mean(axis=1), dtype=np.float32)
            if mono.size == 0 or not np.isfinite(mono).all():
                raise SourceCleanerError(f"invalid audio for {row['_item_id']}")
            normalized, audio_audit = deterministic_resample_mono(
                mono,
                source_sample_rate=int(source_sample_rate),
                target_sample_rate=int(backend.sample_rate),
            )
            mixtures.append(normalized)
            input_audio_audits.append(audio_audit)
        primary_prompts = [canonical_prompts(row)[0] for row in batch]
        paraphrases = [canonical_prompts(row)[1] for row in batch]
        separated = list(
            backend.separate(
                [*mixtures, *mixtures], [*primary_prompts, *paraphrases]
            )
        )
        separator_calls_now += len(separated)
        if len(separated) != 2 * len(batch):
            raise SourceCleanerError(
                f"backend returned {len(separated)} stems for {2 * len(batch)} queries"
            )
        primary_stems = [
            np.asarray(value, dtype=np.float32) for value in separated[: len(batch)]
        ]
        alternate_stems = [
            np.asarray(value, dtype=np.float32) for value in separated[len(batch) :]
        ]
        residuals = [mixture - stem for mixture, stem in zip(mixtures, primary_stems)]
        audio_vectors = np.asarray(
            backend.embed_audio([*primary_stems, *alternate_stems, *residuals]),
            dtype=np.float64,
        )
        if audio_vectors.ndim != 2 or audio_vectors.shape[0] != 3 * len(batch):
            raise SourceCleanerError("backend returned invalid audio embedding shape")
        target_texts = [_canonical_display_name(row) for row in batch]
        all_other_texts = [_other_display_names(row) for row in batch]
        unique_texts = list(
            dict.fromkeys(
                [*target_texts, *(name for names in all_other_texts for name in names)]
            )
        )
        text_matrix = np.asarray(backend.embed_text(unique_texts), dtype=np.float64)
        if text_matrix.ndim != 2 or text_matrix.shape[0] != len(unique_texts):
            raise SourceCleanerError("backend returned invalid text embedding shape")
        text_vectors = dict(zip(unique_texts, text_matrix))
        for index, row in enumerate(batch):
            stem = primary_stems[index]
            alternate = alternate_stems[index]
            mixture = mixtures[index]
            if stem.shape != mixture.shape or alternate.shape != mixture.shape:
                raise SourceCleanerError(
                    f"separator length mismatch for {row['_item_id']}"
                )
            residual = residuals[index]
            metrics = quality_metrics(
                mixture=mixture,
                stem=stem,
                alternate_stem=alternate,
                residual=residual,
                target_text_vector=text_vectors[target_texts[index]],
                other_text_vectors=[
                    text_vectors[name] for name in all_other_texts[index]
                ],
                stem_audio_vector=audio_vectors[index],
                alternate_audio_vector=audio_vectors[len(batch) + index],
                residual_audio_vector=audio_vectors[2 * len(batch) + index],
            )
            gate_result = apply_quality_gate(
                metrics,
                ambiguity_tier=int(row["ambiguity_tier"]),
                gate=config.quality_gate,
            )
            # A nominal tail may be zero-padded by the transport layer when a
            # YouTube-derived clip decodes slightly shorter than its official
            # annotation timeline.  Such a row must remain terminally
            # accounted for, but it can never become training data even when
            # AudioSep finds a semantically similar sound elsewhere in the
            # crop.  This is a transport-validity rejection, not a learned or
            # QA-tuned quality threshold.
            retained_fraction = float(
                row.get("coverage_event_retained_fraction", 1.0)
            )
            metrics["coverage_event_retained_fraction"] = retained_fraction
            if retained_fraction < 1.0 - 1e-9:
                gate_result = dict(gate_result)
                gate_result["accepted"] = False
                gate_result["acceptance_tier"] = "rejected"
                gate_result["transport_validity_failure"] = (
                    "coverage_event_not_fully_decoded"
                )
                for key in ("gold_failures", "silver_failures"):
                    failures = list(gate_result.get(key) or [])
                    if "coverage_event_not_fully_decoded" not in failures:
                        failures.append("coverage_event_not_fully_decoded")
                    gate_result[key] = failures
            record = _prepare_record(
                row=row,
                transaction_hash=transactions[str(row["_item_id"])],
                backend_identity=backend_identity,
                input_audio_audit=input_audio_audits[index],
                sample_rate=int(backend.sample_rate),
                mixture=mixture,
                stem=stem,
                alternate_stem=alternate,
                residual=residual,
                metrics=metrics,
                gate_result=gate_result,
                output_dir=output_dir,
                store_residual=config.store_residual,
            )
            fragment_path = (
                output_dir / "fragments" / f"{record['transaction_sha256']}.json"
            )
            atomic_json(fragment_path, record)
            fragments[str(row["_item_id"])] = record
            processed_now += 1

    ordered = [fragments[str(row["_item_id"])] for row in rows]
    accepted = [row for row in ordered if bool(row["accepted"])]
    rejected = [row for row in ordered if not bool(row["accepted"])]
    atomic_jsonl(output_dir / "source_bank.jsonl", accepted)
    atomic_jsonl(output_dir / "quality_audit.jsonl", ordered)
    source_shards = _write_shards(
        output_dir / "source_bank_shards", accepted, shard_size=config.shard_size
    )
    audit_shards = _write_shards(
        output_dir / "quality_audit_shards", ordered, shard_size=config.shard_size
    )
    quota = _quota_report(
        ordered,
        train_target=config.target_train_per_class,
        eval_target=config.target_eval_per_class,
    )
    atomic_json(output_dir / "per_class_quota.json", quota)
    tsv_columns = (
        "label",
        "input_train_unique_videos",
        "input_eval_unique_videos",
        "accepted_train_unique_videos",
        "accepted_eval_unique_videos",
        "missing_train_unique_videos",
        "missing_eval_unique_videos",
        "quota_ready",
    )
    tsv = "\t".join(tsv_columns) + "\n" + "".join(
        "\t".join(str(row[column]) for column in tsv_columns) + "\n"
        for row in quota["per_class"]
    )
    atomic_bytes(output_dir / "per_class_quota.tsv", tsv.encode("utf-8"))
    acceptance_counts = Counter(str(row["acceptance_tier"]) for row in ordered)
    input_sample_rate_counts = Counter(
        int(row["input_sample_rate"]) for row in ordered
    )
    materialization_route_counts = Counter(
        str(row["materialization_source_route"] or "unspecified")
        for row in ordered
    )
    tier_counts = Counter(int(row["ambiguity_tier"]) for row in ordered)
    accepted_tier_counts = Counter(
        int(row["ambiguity_tier"]) for row in accepted
    )
    receipt = {
        "format": FORMAT,
        "quality_protocol": QUALITY_PROTOCOL,
        "input_manifest_receipts": input_receipts,
        "selected_input_sha256": input_hash,
        "backend": backend_identity,
        "input_normalizer": input_normalizer_identity,
        "input_sample_rate_counts": {
            str(key): value for key, value in sorted(input_sample_rate_counts.items())
        },
        "materialization_source_route_counts": dict(
            sorted(materialization_route_counts.items())
        ),
        "resampling_applied_items": sum(
            bool(row["resampling_applied"]) for row in ordered
        ),
        "maximum_absolute_resample_duration_error_seconds": max(
            (
                abs(float(row["resample_duration_error_seconds"]))
                for row in ordered
            ),
            default=0.0,
        ),
        "quality_gate": asdict(config.quality_gate),
        "processing_order": "ambiguity_tier_0_then_1_then_2_then_3",
        "store_residual": bool(config.store_residual),
        "max_items": config.max_items,
        "input_items": len(ordered),
        "accepted_items": len(accepted),
        "rejected_items": len(rejected),
        "acceptance_counts": dict(sorted(acceptance_counts.items())),
        "input_ambiguity_tier_counts": {
            str(key): value for key, value in sorted(tier_counts.items())
        },
        "accepted_ambiguity_tier_counts": {
            str(key): value for key, value in sorted(accepted_tier_counts.items())
        },
        "resumed_items": len(ordered) - processed_now,
        "processed_items_this_run": processed_now,
        "effective_separator_queries_this_run": separator_calls_now,
        "source_bank_manifest": {
            "path": _portable_path(output_dir / "source_bank.jsonl"),
            "sha256": sha256_file(output_dir / "source_bank.jsonl"),
            "rows": len(accepted),
            "shards": source_shards,
        },
        "quality_audit_manifest": {
            "path": _portable_path(output_dir / "quality_audit.jsonl"),
            "sha256": sha256_file(output_dir / "quality_audit.jsonl"),
            "rows": len(ordered),
            "shards": audit_shards,
        },
        "quota_report": quota,
        "invariants": {
            "source_video_id_preserved": all(
                bool(row["source_video_id"]) for row in ordered
            ),
            "upstream_split_preserved_and_never_reassigned": True,
            "all_intersecting_strong_annotations_preserved": True,
            "residual_is_source_minus_primary_stem": True,
            "residual_artifact_storage_is_explicit": True,
            "qa_answers_used_for_selection": False,
            "downstream_accuracy_used_for_selection": False,
            "same_fixed_quality_thresholds_for_all_splits": True,
            "tier_0_1_processed_before_tier_2_3": True,
            "all_separator_inputs_use_backend_sample_rate": all(
                int(row["separator_input_sample_rate"])
                == int(backend.sample_rate)
                for row in ordered
            ),
            "resampling_is_content_bound_and_timestamp_seconds_preserved": True,
        },
    }
    atomic_json(output_dir / "source_bank_receipt.json", receipt)
    return receipt
