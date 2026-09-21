"""Strict TACOS metadata selection for the QCES v5 real-audio evaluation.

The synthetic QCES benchmark has exact clean stems, while TACOS supplies a
separately sourced real-recording stress test.  This module deliberately stops
at an immutable *annotation packet*: TACOS captions and timestamps are useful
proposals, but they are not treated as unquestioned QCES ground truth.  A
later human-verification step must accept/correct audible events before any
real-data submission gate can pass.

The primary route is CC0-only audio.  This is a scientific/release choice, not
an assertion that the other TACOS recordings are unlicensed: the official
record contains CC-BY, CC-BY-NC, and Sampling+ recordings as well.  Restricting
the headline set to CC0 removes per-record derivative-license ambiguity while
retaining TACOS attribution for its CC-BY-4.0 captions.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


PLAN_FORMAT = "qces_v5_tacos_real_source_plan_v3"
COMPLIANCE_FORMAT = "qces_v5_tacos_real_compliance_v3"
PACKET_FORMAT = "qces_v5_tacos_annotation_packet_v3"
LABEL_SELECTION_FORMAT = "qces_v5_source_selection_v1"
RECORD_ID = 15_379_789
DOI = "10.5281/zenodo.15379789"
DATASET_NAME = "TACOS"
CAPTION_LICENSE_SPDX = "CC-BY-4.0"
CAPTION_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CC0_SPDX = "CC0-1.0"
CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
CREATOR_KEY_PREFIX = "freesound-user-sha256:"
CREATOR_KEY_DOMAIN = b"qces-v5-tacos-freesound-user-v1\0"
CUSTOM_SPLIT_HASH_DOMAIN = b"qces-v5-tacos-custom-creator-split-v2\0"
WINDOW_HASH_DOMAIN = b"qces-v5-tacos-window-start-v2\0"
SELECTION_HASH_DOMAIN = b"qces-v5-tacos-selection-v2\0"
ABSENT_POOL_HASH_DOMAIN = b"qces-v5-tacos-cross-scene-absent-pool-v3\0"
BENCHMARK_SAMPLE_RATE = 32_000
DEFAULT_BENCHMARK_WINDOW_SECONDS = 10.0
SOURCE_DURATION_SAFETY_SECONDS = 0.01
SOURCE_DURATION_SAFETY_SAMPLES = math.ceil(
    SOURCE_DURATION_SAFETY_SECONDS * BENCHMARK_SAMPLE_RATE
)
CUSTOM_DEVELOPMENT_NUMERATOR = 1
CUSTOM_DEVELOPMENT_DENOMINATOR = 5

SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
SEMANTIC_MODEL_WEIGHTS = "model.safetensors"
SEMANTIC_MODEL_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
SEMANTIC_EMBEDDING_IMPLEMENTATION = "transformers_mean_pool_l2_v1"
SEMANTIC_REQUIRED_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)

# Record 15379789 is immutable.  Pinning these values independently of the
# downloaded API JSON prevents a modified descriptor from blessing modified
# inputs.  Values come from the official Zenodo record.
OFFICIAL_FILES: Mapping[str, tuple[int, str]] = {
    "annotations_strong.csv": (4_109_451, "882d5a5a28f59441f4c7b4ed17ebab05"),
    "audio.zip": (1_629_958_981, "a1b11ed7a88d6b95109decb577c128fd"),
    "development_split.csv": (112_843, "bee28197d13e859dbc36021c2a928f07"),
    "annotations_weak.csv": (1_318_097, "1445727f4d470273a7395ddcb9e34b93"),
    "metadata.csv": (8_134_917, "17e8df44f2d217dc7cf4b75767ee5ab3"),
    "test_split.csv": (21_794, "16f5b5eab84753a571e9944739508854"),
}

METADATA_FIELDS = (
    "filename",
    "keywords",
    "freesound_id",
    "sound_link",
    "manufacturer",
    "license",
    "superclass",
    "subclass",
    "title",
    "description",
    "duration",
    "original_samplerate",
    "original_bitdepth",
    "original_file_type",
    "num_downloads",
    "avg_rating",
    "geotag",
    "start_time_s",
    "end_time_s",
    "original_filename",
)
STRONG_FIELDS = ("filename", "text", "onset", "offset")
WEAK_FIELDS = ("filename", "text")
SPLIT_FIELDS = ("filename",)

GENERIC_CAPTIONS = {
    "audio",
    "ambient noise",
    "background noise",
    "no sound",
    "noise",
    "silence",
    "sound",
    "sounds",
}

QUESTION_SLOTS: tuple[Mapping[str, Any], ...] = (
    {
        "slot_id": "after_internal_0",
        "relation": "after",
        "anchor_selector": "verified_rank_0",
        "answer_selector": "verified_rank_1",
        "no_evidence": False,
        "option_order_variant": "forward",
    },
    {
        "slot_id": "after_internal_1",
        "relation": "after",
        "anchor_selector": "verified_rank_1",
        "answer_selector": "verified_rank_2",
        "no_evidence": False,
        "option_order_variant": "forward",
    },
    {
        "slot_id": "before_internal_0",
        "relation": "before",
        "anchor_selector": "verified_rank_2",
        "answer_selector": "verified_rank_1",
        "no_evidence": False,
        "option_order_variant": "forward",
    },
    {
        "slot_id": "before_internal_1",
        "relation": "before",
        "anchor_selector": "verified_rank_3",
        "answer_selector": "verified_rank_2",
        "no_evidence": False,
        "option_order_variant": "forward",
    },
    {
        "slot_id": "first_pair_forward",
        "relation": "first",
        "anchor_selector": "verified_rank_0_and_2",
        "answer_selector": "verified_rank_0",
        "no_evidence": False,
        "option_order_variant": "forward",
    },
    {
        "slot_id": "first_pair_reversed",
        "relation": "first",
        "anchor_selector": "verified_rank_0_and_2",
        "answer_selector": "verified_rank_0",
        "no_evidence": False,
        "option_order_variant": "reversed",
    },
    {
        "slot_id": "after_absent_anchor_no_evidence",
        "relation": "after",
        "anchor_selector": "human_verified_absent_anchor",
        "answer_selector": "no_evidence",
        "no_evidence": True,
        "option_order_variant": "balanced",
    },
    {
        "slot_id": "before_absent_anchor_no_evidence",
        "relation": "before",
        "anchor_selector": "human_verified_absent_anchor",
        "answer_selector": "no_evidence",
        "no_evidence": True,
        "option_order_variant": "balanced",
    },
)


class TacosAuditError(ValueError):
    """Raised when official metadata or the real-set contract is unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies the publisher's exact pin
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_semantic_model_snapshot(path: Path) -> Mapping[str, Any]:
    """Validate the exact offline text encoder used for chain selection.

    The planner never resolves a moving Hub alias.  It accepts only the pinned
    snapshot directory, verifies the weight bytes independently, and records a
    digest of every required local artifact.
    """

    resolved = path.resolve()
    if not path.is_dir() or path.name != SEMANTIC_MODEL_REVISION:
        raise TacosAuditError(
            "semantic model must be the local pinned snapshot directory "
            f"ending in {SEMANTIC_MODEL_REVISION}"
        )
    files: dict[str, Mapping[str, Any]] = {}
    for relative_name in SEMANTIC_REQUIRED_FILES:
        artifact = path / relative_name
        if not artifact.is_file():
            raise TacosAuditError(f"missing semantic model artifact: {artifact}")
        files[relative_name] = {
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
    weights_sha256 = files[SEMANTIC_MODEL_WEIGHTS]["sha256"]
    if weights_sha256 != SEMANTIC_MODEL_SHA256:
        raise TacosAuditError(
            "semantic model weight SHA-256 mismatch: "
            f"expected={SEMANTIC_MODEL_SHA256}, actual={weights_sha256}"
        )

    try:
        pooling = json.loads((path / "1_Pooling/config.json").read_text("utf-8"))
        modules = json.loads((path / "modules.json").read_text("utf-8"))
        sentence_config = json.loads(
            (path / "sentence_bert_config.json").read_text("utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise TacosAuditError("invalid semantic model configuration") from error
    expected_pooling = {
        "word_embedding_dimension": 384,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }
    if pooling != expected_pooling:
        raise TacosAuditError("semantic model pooling configuration mismatch")
    module_types = [row.get("type") for row in modules if isinstance(row, dict)]
    if module_types != [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ]:
        raise TacosAuditError("semantic model module graph mismatch")
    if sentence_config != {"max_seq_length": 256, "do_lower_case": False}:
        raise TacosAuditError("semantic model sentence configuration mismatch")

    fingerprint_payload = {
        "model_id": SEMANTIC_MODEL_ID,
        "revision": SEMANTIC_MODEL_REVISION,
        "embedding_implementation": SEMANTIC_EMBEDDING_IMPLEMENTATION,
        "files": files,
    }
    return {
        **fingerprint_payload,
        "snapshot_path": str(resolved),
        "weights_sha256": weights_sha256,
        "snapshot_fingerprint": canonical_json_sha256(fingerprint_payload),
        "embedding_dimension": 384,
        "max_sequence_length": 256,
        "network_access_required": False,
    }


def encode_semantic_captions(
    captions: Sequence[str],
    *,
    snapshot_path: Path,
    batch_size: int,
) -> tuple[dict[str, tuple[float, ...]], Mapping[str, Any]]:
    """Encode normalized captions on CPU with the pinned ST module graph."""

    if batch_size <= 0:
        raise TacosAuditError("semantic embedding batch size must be positive")
    receipt = dict(validate_semantic_model_snapshot(snapshot_path))
    unique = sorted(set(captions))
    if any(not value or value != normalize_caption(value) for value in unique):
        raise TacosAuditError("semantic captions must be nonempty and normalized")
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise TacosAuditError(
            "semantic selection requires torch and transformers; run the planner "
            "from the declared QCES environment"
        ) from error

    # CPU inference avoids consuming the experiment GPU and fixes the device
    # path used to create this immutable metadata split.
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot_path), local_files_only=True)
    model.to("cpu")
    model.eval()
    encoded: dict[str, tuple[float, ...]] = {}
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            outputs = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(outputs.dtype)
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for caption, vector in zip(batch, pooled.cpu().tolist(), strict=True):
                if len(vector) != 384 or any(not math.isfinite(x) for x in vector):
                    raise TacosAuditError("semantic model emitted an invalid embedding")
                encoded[caption] = tuple(float(value) for value in vector)
    receipt.update(
        {
            "runtime": {
                "torch_version": str(torch.__version__),
                "transformers_version": str(transformers.__version__),
                "device": "cpu",
                "torch_threads": 1,
                "batch_size": batch_size,
            },
            "unique_captions_encoded_↑": len(encoded),
            "qces_method_outputs_used": False,
            "human_labels_used": False,
        }
    )
    return encoded, receipt


def _atomic_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_text(path, text, overwrite=overwrite)


def atomic_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool
) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_text(path, text, overwrite=overwrite)


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise TacosAuditError(
                f"{path.name} fields mismatch: expected={tuple(fields)!r}, "
                f"actual={tuple(reader.fieldnames or ())!r}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise TacosAuditError(f"{path} is empty")
    return rows


def _finite_float(value: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TacosAuditError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise TacosAuditError(f"{context} must be finite")
    return result


def normalize_caption(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def creator_key(name: str) -> str:
    if not name or name != name.strip():
        raise TacosAuditError("manufacturer/creator must be a trimmed string")
    digest = hashlib.sha256(CREATOR_KEY_DOMAIN + name.encode("utf-8")).hexdigest()
    return CREATOR_KEY_PREFIX + digest


def canonical_license(value: str) -> tuple[str, str] | None:
    normalized = value.strip().casefold().replace("http://", "https://")
    normalized = normalized.rstrip("/") + "/"
    if normalized == CC0_URL.casefold():
        return CC0_SPDX, CC0_URL
    if normalized == "https://creativecommons.org/licenses/by/3.0/":
        return "CC-BY-3.0", "https://creativecommons.org/licenses/by/3.0/"
    if normalized == CAPTION_LICENSE_URL.casefold():
        return CAPTION_LICENSE_SPDX, CAPTION_LICENSE_URL
    return None


def _verify_official_file(path: Path, key: str) -> Mapping[str, Any]:
    expected_size, expected_md5 = OFFICIAL_FILES[key]
    if not path.is_file():
        raise TacosAuditError(f"missing official file: {path}")
    size = path.stat().st_size
    if size != expected_size:
        raise TacosAuditError(
            f"{key} size mismatch: expected={expected_size}, actual={size}"
        )
    actual_md5 = md5_file(path)
    if actual_md5 != expected_md5:
        raise TacosAuditError(
            f"{key} MD5 mismatch: expected={expected_md5}, actual={actual_md5}"
        )
    return {
        "path": str(path.resolve()),
        "size_bytes": size,
        "publisher_md5": actual_md5,
        "sha256": sha256_file(path),
    }


def validate_record_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TacosAuditError(f"cannot parse Zenodo record JSON: {path}") from error
    if payload.get("id") != RECORD_ID:
        raise TacosAuditError(f"unexpected Zenodo record id: {payload.get('id')!r}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("doi") != DOI:
        raise TacosAuditError("Zenodo record DOI mismatch")
    files = payload.get("files")
    if not isinstance(files, list):
        raise TacosAuditError("Zenodo record has no file list")
    actual: dict[str, tuple[int, str]] = {}
    for row in files:
        if not isinstance(row, dict):
            raise TacosAuditError("invalid Zenodo file descriptor")
        key = row.get("key")
        checksum = row.get("checksum")
        size = row.get("size")
        if not isinstance(key, str) or not isinstance(checksum, str):
            raise TacosAuditError("invalid Zenodo file key/checksum")
        if not checksum.startswith("md5:"):
            raise TacosAuditError(f"{key} does not use the pinned MD5 descriptor")
        actual[key] = (size, checksum.removeprefix("md5:"))
    if actual != dict(OFFICIAL_FILES):
        raise TacosAuditError("Zenodo record file inventory differs from the pin")
    return {
        "record_id": RECORD_ID,
        "doi": DOI,
        "record_json_path": str(path.resolve()),
        "record_json_sha256": sha256_file(path),
        "created": payload.get("created"),
        "updated": payload.get("updated"),
        "caption_license_spdx": CAPTION_LICENSE_SPDX,
        "caption_license_url": CAPTION_LICENSE_URL,
        "individual_audio_licenses_declared": True,
    }


def load_qces_label_selection(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TacosAuditError(f"cannot parse QCES label selection: {path}") from error
    expected = {
        "schema_version",
        "seen_labels",
        "heldout_labels",
        "nuisance_labels",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise TacosAuditError("QCES label selection fields mismatch")
    if payload.get("schema_version") != LABEL_SELECTION_FORMAT:
        raise TacosAuditError("QCES label selection schema mismatch")
    parsed: dict[str, list[str]] = {}
    for key in ("seen_labels", "heldout_labels", "nuisance_labels"):
        value = payload[key]
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(value) != len(set(value))
        ):
            raise TacosAuditError(f"invalid QCES label list: {key}")
        parsed[key] = list(value)
    if len(parsed["seen_labels"]) < 10:
        raise TacosAuditError("too few seen labels for absent-anchor proposals")
    return {
        **parsed,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


@dataclass(frozen=True)
class Region:
    region_id: str
    caption: str
    onset: float
    offset: float
    eligible: bool
    exclusion_reasons: tuple[str, ...]

    def as_shifted_dict(
        self,
        *,
        window_start: float,
        window_end: float,
        chain_region_ids: frozenset[str],
    ) -> dict[str, Any] | None:
        """Return a window-relative proposal, or ``None`` when disjoint."""

        if self.offset <= window_start or self.onset >= window_end:
            return None
        shifted_onset = max(0.0, self.onset - window_start)
        shifted_offset = min(window_end - window_start, self.offset - window_start)
        truncated = self.onset < window_start or self.offset > window_end
        reasons = list(self.exclusion_reasons)
        if truncated:
            reasons.append("truncated_by_benchmark_window")
        return {
            "region_id": self.region_id,
            "caption": self.caption,
            "onset_seconds": round(shifted_onset, 9),
            "offset_seconds": round(shifted_offset, 9),
            "upstream_clip_onset_seconds": self.onset,
            "upstream_clip_offset_seconds": self.offset,
            "truncated_by_benchmark_window": truncated,
            "selected_chain_member": self.region_id in chain_region_ids,
            "eligible_proposal": self.eligible and not truncated,
            "proposal_exclusion_reasons": reasons,
        }


@dataclass(frozen=True)
class ChainProposal:
    region_ids: tuple[str, str, str, str]
    pairwise_cosines: tuple[float, float, float, float, float, float]
    max_pairwise_cosine: float
    mean_pairwise_cosine: float
    minimum_adjacent_onset_gap: float
    first_onset: float
    last_offset: float
    valid_window_start_sample_low: int
    valid_window_start_sample_high: int
    window_start_sample: int
    window_end_sample: int
    chain_rank_tie_sha256: str
    window_start_sha256: str
    ranked_chain_count: int
    semantically_diverse_chain_count: int

    @property
    def window_start(self) -> float:
        return self.window_start_sample / BENCHMARK_SAMPLE_RATE

    @property
    def window_end(self) -> float:
        return self.window_end_sample / BENCHMARK_SAMPLE_RATE


@dataclass(frozen=True)
class Candidate:
    filename: str
    freesound_id: str
    sound_link: str
    creator_name: str
    creator_id: str
    superclass: str
    subclass: str
    upstream_split: str
    audio_license_spdx: str
    audio_license_url: str
    original_duration: float
    crop_start: float
    crop_end: float
    regions: tuple[Region, ...]
    source_metadata_reasons: tuple[str, ...]
    best_chain: ChainProposal | None = None
    ranked_chain_count: int = 0
    semantically_diverse_chain_count: int = 0

    @property
    def clip_duration(self) -> float:
        return self.crop_end - self.crop_start

    @property
    def usable_region_count(self) -> int:
        return sum(region.eligible for region in self.regions)

    @property
    def suggested_region_ids(self) -> tuple[str, ...]:
        return self.best_chain.region_ids if self.best_chain is not None else ()

    @property
    def overlap_pair_count(self) -> int:
        if self.best_chain is None:
            return 0
        selected = [
            region
            for region in self.regions
            if region.region_id in self.best_chain.region_ids
        ]
        return sum(
            max(left.onset, right.onset) < min(left.offset, right.offset)
            for index, left in enumerate(selected)
            for right in selected[index + 1 :]
        )

    @property
    def nonoverlap_pair_count(self) -> int:
        return 6 - self.overlap_pair_count if self.best_chain is not None else 0

    @property
    def max_concurrent_region_count(self) -> int:
        if self.best_chain is None:
            return 0
        selected_ids = frozenset(self.best_chain.region_ids)
        boundaries = sorted(
            (
                (time, boundary_order, delta)
                for region in self.regions
                if region.region_id in selected_ids
                for time, boundary_order, delta in (
                    (region.onset, 1, 1),
                    (region.offset, 0, -1),
                )
            ),
            key=lambda row: (row[0], row[1]),
        )
        concurrent = 0
        maximum = 0
        for _time, _order, delta in boundaries:
            concurrent += delta
            maximum = max(maximum, concurrent)
        return maximum


def _safe_filename(value: str, context: str) -> str:
    path = PurePosixPath(value)
    if (
        path.name != value
        or path.suffix.casefold() != ".mp3"
        or not path.stem.isdigit()
    ):
        raise TacosAuditError(f"{context} is not a safe numeric MP3 filename")
    return value


def _validate_sound_link(link: str, freesound_id: str, context: str) -> str:
    parsed = urllib.parse.urlparse(link)
    if parsed.scheme != "https" or parsed.hostname != "freesound.org":
        raise TacosAuditError(f"{context} must use https://freesound.org")
    if not parsed.path.endswith(f"/sounds/{freesound_id}/"):
        raise TacosAuditError(f"{context} does not match freesound_id")
    if parsed.query or parsed.fragment:
        raise TacosAuditError(f"{context} must not contain query/fragment")
    return link


def valid_window_start_sample_interval(
    regions: Sequence[Region],
    *,
    clip_duration: float,
    window_seconds: float,
) -> tuple[int, int] | None:
    """Return every 32 kHz start sample that fully contains ``regions``."""

    if len(regions) != 4:
        raise TacosAuditError("benchmark-window interval requires exactly four regions")
    if not (
        math.isfinite(clip_duration)
        and math.isfinite(window_seconds)
        and window_seconds > 0.0
    ):
        raise TacosAuditError("invalid benchmark window duration")
    window_samples = round(window_seconds * BENCHMARK_SAMPLE_RATE)
    metadata_duration_floor_samples = math.floor(clip_duration * BENCHMARK_SAMPLE_RATE)
    conservative_source_frames = (
        metadata_duration_floor_samples - SOURCE_DURATION_SAFETY_SAMPLES
    )
    if conservative_source_frames < window_samples:
        return None
    # The ceil/floor directions are intentional: every selected proposal must
    # be fully contained after quantizing the crop to 32 kHz sample indices.
    low = max(
        0,
        max(math.ceil(region.offset * BENCHMARK_SAMPLE_RATE) for region in regions)
        - window_samples,
    )
    high = min(
        min(math.floor(region.onset * BENCHMARK_SAMPLE_RATE) for region in regions),
        conservative_source_frames - window_samples,
    )
    return (low, high) if low <= high else None


def _rounded_cosine(
    left: Sequence[float], right: Sequence[float], *, decimals: int
) -> float:
    if len(left) != len(right) or not left:
        raise TacosAuditError("semantic embedding dimensions do not match")
    value = sum(a * b for a, b in zip(left, right, strict=True))
    if not math.isfinite(value):
        raise TacosAuditError("nonfinite semantic cosine")
    return round(float(value), decimals)


def rank_candidate_chains(
    candidate: Candidate,
    *,
    embeddings: Mapping[str, Sequence[float]],
    seed: int,
    minimum_onset_gap_seconds: float,
    benchmark_window_seconds: float,
    maximum_pairwise_cosine: float,
    cosine_round_decimals: int,
) -> Candidate:
    """Attach the best four-region semantic chain and its fixed 10 s window."""

    if not 0.0 <= maximum_pairwise_cosine <= 1.0:
        raise TacosAuditError("maximum pairwise cosine must lie in [0, 1]")
    if not 0 <= cosine_round_decimals <= 12:
        raise TacosAuditError("cosine rounding decimals must lie in [0, 12]")
    window_samples = round(benchmark_window_seconds * BENCHMARK_SAMPLE_RATE)
    eligible = sorted(
        (region for region in candidate.regions if region.eligible),
        key=lambda item: (item.onset, item.offset, item.region_id),
    )
    chain_count = 0
    diverse_count = 0
    ranked: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    for regions in itertools.combinations(eligible, 4):
        normalized = tuple(normalize_caption(region.caption) for region in regions)
        if len(set(normalized)) != 4:
            continue
        adjacent_gaps = tuple(
            right.onset - left.onset for left, right in zip(regions, regions[1:])
        )
        if min(adjacent_gaps) < minimum_onset_gap_seconds:
            continue
        valid_interval = valid_window_start_sample_interval(
            regions,
            clip_duration=candidate.clip_duration,
            window_seconds=benchmark_window_seconds,
        )
        if valid_interval is None:
            continue
        chain_count += 1
        try:
            vectors = tuple(embeddings[text] for text in normalized)
        except KeyError as error:
            raise TacosAuditError(
                f"missing semantic embedding for caption: {error.args[0]!r}"
            ) from error
        cosines = tuple(
            _rounded_cosine(vectors[i], vectors[j], decimals=cosine_round_decimals)
            for i, j in itertools.combinations(range(4), 2)
        )
        maximum = max(cosines)
        if maximum > maximum_pairwise_cosine:
            continue
        diverse_count += 1
        mean = round(sum(cosines) / len(cosines), cosine_round_decimals)
        region_ids = tuple(region.region_id for region in regions)
        tie_hash = hashlib.sha256(
            WINDOW_HASH_DOMAIN
            + b"chain-rank\0"
            + (f"{seed}\0{candidate.filename}\0" + "\0".join(region_ids)).encode(
                "utf-8"
            )
        ).hexdigest()
        score = (maximum, mean, tie_hash)
        ranked.append(
            (
                score,
                {
                    "region_ids": region_ids,
                    "cosines": cosines,
                    "maximum": maximum,
                    "mean": mean,
                    "minimum_gap": min(adjacent_gaps),
                    "first_onset": min(region.onset for region in regions),
                    "last_offset": max(region.offset for region in regions),
                    "valid_low": valid_interval[0],
                    "valid_high": valid_interval[1],
                    "tie_hash": tie_hash,
                },
            )
        )
    if not ranked:
        return replace(
            candidate,
            best_chain=None,
            ranked_chain_count=chain_count,
            semantically_diverse_chain_count=diverse_count,
        )
    chosen = min(ranked, key=lambda row: row[0])[1]
    region_ids = chosen["region_ids"]
    cosines = chosen["cosines"]
    if len(region_ids) != 4 or len(cosines) != 6:
        raise AssertionError("four-region chain invariant failed")
    start_digest = hashlib.sha256(
        WINDOW_HASH_DOMAIN
        + b"hash-uniform-start\0"
        + (f"{seed}\0{candidate.filename}\0" + "\0".join(region_ids)).encode("utf-8")
    ).digest()
    valid_low = chosen["valid_low"]
    valid_high = chosen["valid_high"]
    window_start_sample = valid_low + int.from_bytes(
        start_digest, byteorder="big", signed=False
    ) % (valid_high - valid_low + 1)
    proposal = ChainProposal(
        region_ids=region_ids,
        pairwise_cosines=cosines,
        max_pairwise_cosine=chosen["maximum"],
        mean_pairwise_cosine=chosen["mean"],
        minimum_adjacent_onset_gap=chosen["minimum_gap"],
        first_onset=chosen["first_onset"],
        last_offset=chosen["last_offset"],
        valid_window_start_sample_low=valid_low,
        valid_window_start_sample_high=valid_high,
        window_start_sample=window_start_sample,
        window_end_sample=window_start_sample + window_samples,
        chain_rank_tie_sha256=chosen["tie_hash"],
        window_start_sha256=start_digest.hex(),
        ranked_chain_count=chain_count,
        semantically_diverse_chain_count=diverse_count,
    )
    return replace(
        candidate,
        best_chain=proposal,
        ranked_chain_count=chain_count,
        semantically_diverse_chain_count=diverse_count,
    )


def _parse_inputs(
    *,
    metadata_path: Path,
    strong_path: Path,
    weak_path: Path,
    development_split_path: Path,
    test_split_path: Path,
    minimum_region_seconds: float,
) -> tuple[dict[str, Candidate], Mapping[str, tuple[str, ...]], Mapping[str, Any]]:
    metadata_rows = _read_csv(metadata_path, METADATA_FIELDS)
    strong_rows = _read_csv(strong_path, STRONG_FIELDS)
    weak_rows = _read_csv(weak_path, WEAK_FIELDS)
    development_rows = _read_csv(development_split_path, SPLIT_FIELDS)
    test_rows = _read_csv(test_split_path, SPLIT_FIELDS)

    metadata: dict[str, dict[str, Any]] = {}
    license_counts: Counter[str] = Counter()
    unknown_license_count = 0
    source_crop_exceeds_original_count = 0
    for row_index, row in enumerate(metadata_rows):
        context = f"metadata row {row_index}"
        filename = _safe_filename(row["filename"], f"{context}.filename")
        if filename in metadata:
            raise TacosAuditError(f"duplicate metadata filename: {filename}")
        freesound_id = row["freesound_id"]
        if freesound_id != PurePosixPath(filename).stem:
            raise TacosAuditError(f"{context} filename/freesound_id mismatch")
        link = _validate_sound_link(row["sound_link"], freesound_id, context)
        original_duration = _finite_float(row["duration"], f"{context}.duration")
        crop_start = _finite_float(row["start_time_s"], f"{context}.start_time_s")
        crop_end = _finite_float(row["end_time_s"], f"{context}.end_time_s")
        if not (original_duration > 0.0 and 0.0 <= crop_start < crop_end):
            raise TacosAuditError(f"{context} has invalid source/crop duration")
        source_metadata_reasons: list[str] = []
        # The immutable TACOS metadata contains a small number of source
        # duration/crop disagreements (including a few multi-second cases).
        # Preserve and count the defect, but never select such a row.  The
        # supplied MP3 duration is verified independently after extraction.
        if crop_end > original_duration + 0.05:
            source_metadata_reasons.append("crop_end_exceeds_original_duration")
            source_crop_exceeds_original_count += 1
        license_info = canonical_license(row["license"])
        if license_info is None:
            unknown_license_count += int(
                "creativecommons.org" not in row["license"].casefold()
            )
            license_spdx = "NOT-ALLOWLISTED"
            license_url = row["license"].strip()
        else:
            license_spdx, license_url = license_info
        license_counts[license_spdx] += 1
        creator_name = row["manufacturer"]
        metadata[filename] = {
            "filename": filename,
            "freesound_id": freesound_id,
            "sound_link": link,
            "creator_name": creator_name,
            "creator_id": creator_key(creator_name),
            "superclass": row["superclass"].strip(),
            "subclass": row["subclass"].strip(),
            "original_duration": original_duration,
            "crop_start": crop_start,
            "crop_end": crop_end,
            "license_spdx": license_spdx,
            "license_url": license_url,
            "source_metadata_reasons": tuple(source_metadata_reasons),
        }
        if not metadata[filename]["superclass"] or not metadata[filename]["subclass"]:
            raise TacosAuditError(f"{context} lacks superclass/subclass")

    annotations: defaultdict[str, list[tuple[float, float, str, tuple[str, ...]]]] = (
        defaultdict(list)
    )
    invalid_strong_interval_count = 0
    for row_index, row in enumerate(strong_rows):
        filename = _safe_filename(
            row["filename"], f"strong annotation row {row_index}.filename"
        )
        if filename not in metadata:
            raise TacosAuditError(f"annotation references unknown file: {filename}")
        text = row["text"].strip()
        if not text:
            raise TacosAuditError(f"empty strong caption for {filename}")
        onset = _finite_float(row["onset"], f"{filename}.onset")
        offset = _finite_float(row["offset"], f"{filename}.offset")
        clip_duration = (
            metadata[filename]["crop_end"] - metadata[filename]["crop_start"]
        )
        interval_reasons: list[str] = []
        if onset < 0.0:
            interval_reasons.append("negative_onset")
        if offset <= onset:
            interval_reasons.append("nonpositive_interval")
        if offset > clip_duration + 1e-4:
            interval_reasons.append("offset_exceeds_clip_duration")
        invalid_strong_interval_count += int(bool(interval_reasons))
        annotations[filename].append((onset, offset, text, tuple(interval_reasons)))

    if set(annotations) != set(metadata):
        missing = sorted(set(metadata) - set(annotations))[:5]
        extra = sorted(set(annotations) - set(metadata))[:5]
        raise TacosAuditError(
            f"strong annotation coverage mismatch: missing={missing}, extra={extra}"
        )

    weak_counts: Counter[str] = Counter()
    for row_index, row in enumerate(weak_rows):
        filename = _safe_filename(row["filename"], f"weak row {row_index}.filename")
        if filename not in metadata:
            raise TacosAuditError(f"weak caption references unknown file: {filename}")
        if not row["text"].strip():
            raise TacosAuditError(f"empty weak caption for {filename}")
        weak_counts[filename] += 1
    if set(weak_counts) != set(metadata):
        raise TacosAuditError("weak caption coverage does not match metadata")

    def split_ids(rows: Sequence[Mapping[str, str]], name: str) -> tuple[str, ...]:
        values = tuple(_safe_filename(row["filename"], name) for row in rows)
        if len(values) != len(set(values)):
            raise TacosAuditError(f"duplicate filename in {name}")
        unknown = set(values) - set(metadata)
        if unknown:
            raise TacosAuditError(
                f"{name} contains unknown files: {sorted(unknown)[:5]}"
            )
        return values

    development_ids = split_ids(development_rows, "development split")
    test_ids = split_ids(test_rows, "test split")
    overlap = set(development_ids) & set(test_ids)
    if overlap:
        raise TacosAuditError(f"official split overlap: {sorted(overlap)[:5]}")
    if set(development_ids) | set(test_ids) != set(metadata):
        raise TacosAuditError("official splits do not partition metadata exactly")

    upstream_split = {
        **{filename: "development" for filename in development_ids},
        **{filename: "test" for filename in test_ids},
    }
    candidates: dict[str, Candidate] = {}
    for filename, source in metadata.items():
        region_rows = sorted(
            annotations[filename], key=lambda item: (item[0], item[1], item[2])
        )
        regions: list[Region] = []
        for index, (onset, offset, caption, interval_reasons) in enumerate(region_rows):
            reasons: list[str] = list(interval_reasons)
            normalized = normalize_caption(caption)
            if offset - onset < minimum_region_seconds:
                reasons.append("shorter_than_minimum_region")
            if len(normalized) < 4:
                reasons.append("caption_too_short")
            if normalized in GENERIC_CAPTIONS:
                reasons.append("generic_caption")
            regions.append(
                Region(
                    region_id=f"region_{index:03d}",
                    caption=caption,
                    onset=onset,
                    offset=offset,
                    eligible=not reasons,
                    exclusion_reasons=tuple(reasons),
                )
            )
        candidates[filename] = Candidate(
            filename=filename,
            freesound_id=source["freesound_id"],
            sound_link=source["sound_link"],
            creator_name=source["creator_name"],
            creator_id=source["creator_id"],
            superclass=source["superclass"],
            subclass=source["subclass"],
            upstream_split=upstream_split[filename],
            audio_license_spdx=source["license_spdx"],
            audio_license_url=source["license_url"],
            original_duration=source["original_duration"],
            crop_start=source["crop_start"],
            crop_end=source["crop_end"],
            regions=tuple(regions),
            source_metadata_reasons=source["source_metadata_reasons"],
        )

    split_map = {"development": development_ids, "test": test_ids}
    diagnostics = {
        "metadata_rows_↑": len(metadata_rows),
        "strong_annotation_rows_↑": len(strong_rows),
        "weak_annotation_rows_↑": len(weak_rows),
        "development_scenes_↑": len(development_ids),
        "test_scenes_↑": len(test_ids),
        "official_split_overlap_↓": 0,
        "unknown_license_rows_↓": unknown_license_count,
        "known_nonallowlisted_license_rows_↓": license_counts["NOT-ALLOWLISTED"],
        "source_crop_duration_defects_↓": source_crop_exceeds_original_count,
        "invalid_strong_intervals_↓": invalid_strong_interval_count,
        "license_counts": dict(sorted(license_counts.items())),
    }
    return candidates, split_map, diagnostics


def _stable_hash(seed: int, split_name: str, filename: str) -> str:
    return hashlib.sha256(
        SELECTION_HASH_DOMAIN + f"{seed}\0{split_name}\0{filename}".encode("utf-8")
    ).hexdigest()


def assign_creator_hash_rank_partitions(
    creator_ids: Iterable[str],
) -> tuple[dict[str, Mapping[str, Any]], str]:
    """Create an exact 20/80 creator split by domain-separated hash rank."""

    unique = sorted(set(creator_ids))
    if not unique or any(not value for value in unique):
        raise TacosAuditError("creator hash-rank split requires nonempty creator IDs")
    ranked = sorted(
        (
            hashlib.sha256(
                CUSTOM_SPLIT_HASH_DOMAIN + creator_id.encode("utf-8")
            ).hexdigest(),
            creator_id,
        )
        for creator_id in unique
    )
    development_count = (
        len(ranked) * CUSTOM_DEVELOPMENT_NUMERATOR
    ) // CUSTOM_DEVELOPMENT_DENOMINATOR
    if development_count <= 0 or development_count >= len(ranked):
        raise TacosAuditError("creator pool is too small for the frozen 20/80 split")
    assignments: dict[str, Mapping[str, Any]] = {}
    fingerprint_rows: list[Mapping[str, Any]] = []
    for rank, (digest, creator_id) in enumerate(ranked):
        partition = "real_dev" if rank < development_count else "real_test"
        row = {
            "creator_id": creator_id,
            "partition": partition,
            "hash_rank": rank,
            "sha256": digest,
        }
        assignments[creator_id] = row
        fingerprint_rows.append(row)
    return assignments, canonical_json_sha256(fingerprint_rows)


def select_balanced_candidates(
    candidates: Sequence[Candidate],
    *,
    count: int,
    seed: int,
    split_name: str,
    forbidden_creator_ids: Iterable[str] = (),
    overlap_fraction: float = 0.5,
) -> list[Candidate]:
    """Select a deterministic class-balanced, one-creator-per-scene subset."""

    if count <= 0:
        raise TacosAuditError("selection count must be positive")
    if not 0.0 <= overlap_fraction <= 1.0:
        raise TacosAuditError("overlap_fraction must lie in [0, 1]")
    forbidden = set(forbidden_creator_ids)
    pool = [
        candidate
        for candidate in candidates
        if candidate.creator_id not in forbidden
        and candidate.best_chain is not None
        and len(candidate.suggested_region_ids) == 4
        and not candidate.source_metadata_reasons
    ]
    selected: list[Candidate] = []
    used_creators = set(forbidden)
    class_counts: Counter[str] = Counter()
    overlap_selected = 0
    while len(selected) < count:
        available = [item for item in pool if item.creator_id not in used_creators]
        if not available:
            raise TacosAuditError(
                f"cannot select {count} unique creators for {split_name}; "
                f"selected only {len(selected)}"
            )
        target_overlap = round(overlap_fraction * (len(selected) + 1))
        prefer_overlap = overlap_selected < target_overlap

        def score(item: Candidate) -> tuple[Any, ...]:
            if item.best_chain is None:
                raise AssertionError("selected candidate has no semantic chain")
            has_overlap = item.overlap_pair_count > 0
            return (
                class_counts[item.subclass],
                int(has_overlap != prefer_overlap),
                item.best_chain.max_pairwise_cosine,
                item.best_chain.mean_pairwise_cosine,
                _stable_hash(seed, split_name, item.filename),
            )

        chosen = min(available, key=score)
        selected.append(chosen)
        used_creators.add(chosen.creator_id)
        class_counts[chosen.subclass] += 1
        overlap_selected += int(chosen.overlap_pair_count > 0)
    return selected


def build_cross_scene_absent_anchor_pools(
    candidates: Sequence[Candidate],
    *,
    scene_partitions: Mapping[str, str],
    scene_tiers: Mapping[str, str],
    seed: int,
    candidates_per_scene: int = 6,
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    """Build negative anchors only from positive captions in other scenes."""

    if candidates_per_scene < 6:
        raise TacosAuditError(
            "each scene requires at least six absent-anchor candidates"
        )
    scene_candidates: dict[str, Candidate] = {}
    raw_caption_support: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for candidate in candidates:
        if candidate.best_chain is None or len(candidate.best_chain.region_ids) != 4:
            raise TacosAuditError(
                "absent-anchor support requires exact four-region chains"
            )
        scene_id = f"tacos_{candidate.freesound_id}"
        if scene_id in scene_candidates:
            raise TacosAuditError(
                f"duplicate selected scene for absent pool: {scene_id}"
            )
        partition = scene_partitions.get(scene_id)
        if partition not in {"real_dev", "real_test"}:
            raise TacosAuditError(f"missing/invalid selection partition for {scene_id}")
        tier = scene_tiers.get(scene_id)
        if tier not in {"core", "reserve"}:
            raise TacosAuditError(f"missing/invalid selection tier for {scene_id}")
        scene_candidates[scene_id] = candidate
        selected_ids = frozenset(candidate.best_chain.region_ids)
        selected_regions = [
            region for region in candidate.regions if region.region_id in selected_ids
        ]
        if len(selected_regions) != 4:
            raise TacosAuditError(f"selected chain region lookup failed for {scene_id}")
        for region in selected_regions:
            if tier == "core":
                raw_caption_support[(partition, region.caption)].add(scene_id)

    if set(scene_partitions) != set(scene_candidates) or set(scene_tiers) != set(
        scene_candidates
    ):
        raise TacosAuditError("absent-anchor scene partition inventory mismatch")

    support_rows = [
        {
            "selection_partition": partition,
            "support_tier": "core",
            "caption": caption,
            "selected_chain_support_scene_ids": sorted(scene_ids),
            "selected_chain_support_scene_count_↑": len(scene_ids),
        }
        for (partition, caption), scene_ids in sorted(raw_caption_support.items())
    ]
    support_index_fingerprint = canonical_json_sha256(support_rows)
    pools: dict[str, Mapping[str, Any]] = {}
    for scene_id, candidate in sorted(scene_candidates.items()):
        if candidate.best_chain is None:
            raise AssertionError("selected candidate lost its chain")
        partition = scene_partitions[scene_id]
        window_start = candidate.best_chain.window_start
        window_end = candidate.best_chain.window_end
        current_scene_captions = [
            normalize_caption(region.caption)
            for region in candidate.regions
            if region.offset > window_start and region.onset < window_end
        ]
        ranked: list[tuple[str, str, tuple[str, ...]]] = []
        for (
            support_partition,
            caption,
        ), support_scene_ids in raw_caption_support.items():
            if support_partition != partition:
                continue
            normalized = normalize_caption(caption)
            if any(
                normalized == current
                or normalized in current
                or (len(current) >= 3 and current in normalized)
                for current in current_scene_captions
            ):
                continue
            cross_scene_support = tuple(sorted(support_scene_ids - {scene_id}))
            if not cross_scene_support:
                continue
            rank_sha256 = hashlib.sha256(
                ABSENT_POOL_HASH_DOMAIN
                + f"{seed}\0{scene_id}\0{caption}".encode("utf-8")
            ).hexdigest()
            ranked.append((rank_sha256, caption, cross_scene_support))
        ranked.sort(key=lambda row: (row[0], row[1]))
        entries: list[dict[str, Any]] = []
        used_primary_support_scenes: set[str] = set()
        for rank_sha256, caption, support_scene_ids in ranked:
            support_order = sorted(
                support_scene_ids,
                key=lambda support_scene_id: hashlib.sha256(
                    ABSENT_POOL_HASH_DOMAIN
                    + b"primary-support\0"
                    + (f"{seed}\0{scene_id}\0{caption}\0{support_scene_id}").encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )
            primary_support_scene_id = next(
                (
                    support_scene_id
                    for support_scene_id in support_order
                    if support_scene_id not in used_primary_support_scenes
                ),
                None,
            )
            if primary_support_scene_id is None:
                continue
            entries.append(
                {
                    "caption": caption,
                    "selection_partition": partition,
                    "support_tier": "core",
                    "rank_sha256": rank_sha256,
                    "cross_scene_support_scene_ids": list(support_scene_ids),
                    "cross_scene_support_count_↑": len(support_scene_ids),
                    "primary_support_scene_id": primary_support_scene_id,
                }
            )
            used_primary_support_scenes.add(primary_support_scene_id)
            if len(entries) == candidates_per_scene:
                break
        if len(entries) < candidates_per_scene:
            raise TacosAuditError(
                f"too few distinct core support scenes for {scene_id}: "
                f"selected={len(entries)}, required={candidates_per_scene}"
            )
        pools[scene_id] = {
            "selection_partition": partition,
            "absent_anchor_candidates": [entry["caption"] for entry in entries],
            "cross_scene_support": entries,
            "distinct_primary_support_scene_count_↑": len(used_primary_support_scenes),
            "scene_pool_fingerprint": canonical_json_sha256(entries),
        }

    pool_fingerprint = canonical_json_sha256(
        [
            {"scene_id": scene_id, **dict(payload)}
            for scene_id, payload in sorted(pools.items())
        ]
    )
    selected_negative_captions = {
        caption
        for payload in pools.values()
        for caption in payload["absent_anchor_candidates"]
    }
    unsupported = sum(
        not entry["cross_scene_support_scene_ids"]
        for payload in pools.values()
        for entry in payload["cross_scene_support"]
    )
    distinct_support_violations = sum(
        payload["distinct_primary_support_scene_count_↑"] != candidates_per_scene
        for payload in pools.values()
    )
    noncore_support_violations = sum(
        entry["support_tier"] != "core"
        or any(
            scene_tiers[scene_id] != "core"
            for scene_id in entry["cross_scene_support_scene_ids"]
        )
        for payload in pools.values()
        for entry in payload["cross_scene_support"]
    )
    audit = {
        "source": "other_selected_chain_captions",
        "hash_domain_hex": ABSENT_POOL_HASH_DOMAIN.hex(),
        "candidates_per_scene": candidates_per_scene,
        "scene_count_↑": len(pools),
        "selected_partition_caption_pairs_↑": len(raw_caption_support),
        "selected_positive_raw_caption_vocabulary_↑": len(
            {caption for _partition, caption in raw_caption_support}
        ),
        "selected_negative_raw_caption_vocabulary_↑": len(selected_negative_captions),
        "negative_candidate_instances_↑": len(pools) * candidates_per_scene,
        "unsupported_negative_captions_↓": unsupported,
        "cross_partition_support_violations_↓": 0,
        "noncore_support_violations_↓": noncore_support_violations,
        "non_distinct_primary_support_scene_sets_↓": (distinct_support_violations),
        "core_support_real_dev_scenes_↑": sum(
            tier == "core" and scene_partitions[scene_id] == "real_dev"
            for scene_id, tier in scene_tiers.items()
        ),
        "core_support_real_test_scenes_↑": sum(
            tier == "core" and scene_partitions[scene_id] == "real_test"
            for scene_id, tier in scene_tiers.items()
        ),
        "negative_positive_vocabulary_support_fraction_↑": (
            1.0 if selected_negative_captions else 0.0
        ),
        "support_index_fingerprint": support_index_fingerprint,
        "pool_fingerprint": pool_fingerprint,
        "exact_raw_captions_preserved": True,
        "qces_label_selection_used": False,
        "human_labels_used": False,
        "qces_method_outputs_used": False,
    }
    if unsupported or distinct_support_violations or noncore_support_violations:
        raise TacosAuditError("cross-scene absent pool violates support contract")
    return pools, audit


def _scene_packet(
    candidate: Candidate,
    *,
    partition: str,
    tier: str,
    ordinal: int,
    absent_anchor_pool: Mapping[str, Any],
    absent_support_index_fingerprint: str,
    creator_partition_hash: str,
    creator_partition_rank: int,
    maximum_pairwise_cosine: float,
    cosine_round_decimals: int,
    benchmark_window_seconds: float,
) -> dict[str, Any]:
    if candidate.best_chain is None:
        raise TacosAuditError("cannot packet a candidate without a semantic chain")
    absent_anchor_candidates = absent_anchor_pool.get("absent_anchor_candidates")
    absent_anchor_support = absent_anchor_pool.get("cross_scene_support")
    if (
        absent_anchor_pool.get("selection_partition") != partition
        or not isinstance(absent_anchor_candidates, list)
        or len(absent_anchor_candidates) < 6
        or not isinstance(absent_anchor_support, list)
        or len(absent_anchor_support) != len(absent_anchor_candidates)
        or any(
            not isinstance(row, dict)
            or not row.get("cross_scene_support_scene_ids")
            or row.get("selection_partition") != partition
            or row.get("support_tier") != "core"
            or row.get("primary_support_scene_id")
            not in row.get("cross_scene_support_scene_ids", [])
            for row in absent_anchor_support
        )
        or len(
            {
                row["primary_support_scene_id"]
                for row in absent_anchor_support
                if isinstance(row, dict) and "primary_support_scene_id" in row
            }
        )
        != len(absent_anchor_candidates)
    ):
        raise TacosAuditError("invalid cross-scene absent-anchor pool")
    chain = candidate.best_chain
    window_start = chain.window_start
    window_end = chain.window_end
    selected_ids = frozenset(chain.region_ids)
    shifted_regions = [
        shifted
        for region in candidate.regions
        if (
            shifted := region.as_shifted_dict(
                window_start=window_start,
                window_end=window_end,
                chain_region_ids=selected_ids,
            )
        )
        is not None
    ]
    selected_rows = [row for row in shifted_regions if row["selected_chain_member"]]
    if (
        len(selected_rows) != 4
        or len(selected_ids) != 4
        or any(row["truncated_by_benchmark_window"] for row in selected_rows)
    ):
        raise TacosAuditError(
            "selected four-region chain is not fully inside its window"
        )
    scene_id = f"tacos_{candidate.freesound_id}"
    return {
        "schema_version": PACKET_FORMAT,
        "scene_id": scene_id,
        "selection_partition": partition,
        "selection_tier": tier,
        "selection_ordinal": ordinal,
        "audio": {
            # Members live at the root of the official ZIP.  The upstream
            # loader refers to ``audio/<name>`` only after extraction into a
            # directory named ``audio``.
            "archive_member": candidate.filename,
            "benchmark_window_start_sample_32k": chain.window_start_sample,
            "benchmark_window_end_sample_32k": chain.window_end_sample,
            "benchmark_sample_rate_hz": BENCHMARK_SAMPLE_RATE,
            "local_path": None,
            "local_sha256": None,
            "verified_audio_properties": None,
        },
        "source": {
            "dataset": DATASET_NAME,
            "zenodo_record_id": RECORD_ID,
            "doi": DOI,
            "filename": candidate.filename,
            "freesound_id": candidate.freesound_id,
            "source_url": candidate.sound_link,
            "creator_name": candidate.creator_name,
            "creator_id": candidate.creator_id,
            "audio_license_spdx": CC0_SPDX,
            "audio_license_url": CC0_URL,
            "caption_license_spdx": CAPTION_LICENSE_SPDX,
            "caption_license_url": CAPTION_LICENSE_URL,
            "superclass": candidate.superclass,
            "subclass": candidate.subclass,
            "upstream_tacos_split": candidate.upstream_split,
            "custom_qces_partition": partition,
            "custom_creator_partition_sha256": creator_partition_hash,
            "custom_creator_partition_hash_rank": creator_partition_rank,
            "original_duration_seconds": candidate.original_duration,
            "upstream_clip_interval_in_original_source_seconds": [
                candidate.crop_start,
                candidate.crop_end,
            ],
            "upstream_clip_duration_seconds": candidate.clip_duration,
            "benchmark_window_interval_in_upstream_clip_seconds": [
                window_start,
                window_end,
            ],
            "source_crop_interval_seconds": [
                round(candidate.crop_start + window_start, 9),
                round(candidate.crop_start + window_end, 9),
            ],
            "clip_duration_seconds": benchmark_window_seconds,
            "gen_ai_preference_snapshot": {
                "status": "not_available_in_pinned_tacos_metadata",
                "used_for_training": False,
                "evaluation_only": True,
            },
        },
        "benchmark_window": {
            "duration_seconds": benchmark_window_seconds,
            "sample_rate_hz": BENCHMARK_SAMPLE_RATE,
            "source_duration_safety_seconds": SOURCE_DURATION_SAFETY_SECONDS,
            "source_duration_safety_samples": SOURCE_DURATION_SAFETY_SAMPLES,
            "metadata_duration_floor_samples": math.floor(
                candidate.clip_duration * BENCHMARK_SAMPLE_RATE
            ),
            "conservative_source_frame_budget": (
                math.floor(candidate.clip_duration * BENCHMARK_SAMPLE_RATE)
                - SOURCE_DURATION_SAFETY_SAMPLES
            ),
            "start_sample": chain.window_start_sample,
            "end_sample": chain.window_end_sample,
            "start_seconds_in_upstream_clip": window_start,
            "end_seconds_in_upstream_clip": window_end,
            "valid_start_sample_interval_inclusive": [
                chain.valid_window_start_sample_low,
                chain.valid_window_start_sample_high,
            ],
            "selection_rule": (
                "rank eligible four-region chains by rounded maximum cosine, "
                "rounded mean cosine, then seeded SHA-256; choose a sample-"
                "quantized hash-uniform start over the selected chain's full-"
                "containment interval"
            ),
            "valid_start_formula": (
                "low=max(0,max(ceil(offset*32000))-320000); "
                "conservative_frames=floor(duration*32000)-320; "
                "high=min(min(floor(onset*32000)),conservative_frames-320000); "
                "start=low+SHA256(seed,source,chain) mod (high-low+1)"
            ),
            "construction_is_proposal_informed": True,
            "runtime_or_question_specific": False,
            "window_start_sha256": chain.window_start_sha256,
            "selected_before_human_annotation": True,
            "human_labels_used": False,
            "qces_method_outputs_used": False,
        },
        "proposal_regions": shifted_regions,
        "suggested_distinct_onset_chain": list(chain.region_ids),
        "semantic_chain": {
            "region_ids": list(chain.region_ids),
            "pairwise_cosines": list(chain.pairwise_cosines),
            "max_pairwise_cosine_↓": chain.max_pairwise_cosine,
            "mean_pairwise_cosine_↓": chain.mean_pairwise_cosine,
            "maximum_allowed_pairwise_cosine": maximum_pairwise_cosine,
            "cosine_round_decimals": cosine_round_decimals,
            "chain_rank_tie_sha256": chain.chain_rank_tie_sha256,
            "exact_chain_size": 4,
        },
        "proposal_diagnostics": {
            "usable_region_count_↑": candidate.usable_region_count,
            "window_intersecting_proposal_count_↑": len(shifted_regions),
            "selected_semantically_diverse_regions_↑": len(selected_rows),
            "overlap_pair_count_↑": candidate.overlap_pair_count,
            "pairwise_overlap_ratio_↑": candidate.overlap_pair_count / 6.0,
            "nonoverlap_pair_count_↑": candidate.nonoverlap_pair_count,
            "max_concurrent_selected_regions_↑": (
                candidate.max_concurrent_region_count
            ),
            "valid_window_start_samples_↑": (
                chain.valid_window_start_sample_high
                - chain.valid_window_start_sample_low
                + 1
            ),
            "ranked_four_region_chains_↑": chain.ranked_chain_count,
            "semantically_diverse_four_region_chains_↑": (
                chain.semantically_diverse_chain_count
            ),
            "minimum_adjacent_onset_gap_seconds_↑": (chain.minimum_adjacent_onset_gap),
            "source_metadata_defects_↓": len(candidate.source_metadata_reasons),
        },
        "question_contract": {
            "questions_per_accepted_scene": len(QUESTION_SLOTS),
            "minimum_verified_distinct_events": 4,
            "slots": [dict(slot) for slot in QUESTION_SLOTS],
            "answers_options_and_phrases_created_only_after_human_verification": True,
            "absent_anchor_candidates": list(absent_anchor_candidates),
            "absent_anchor_pool_source": "other_selected_chain_captions",
            "absent_anchor_cross_scene_support": absent_anchor_support,
            "absent_anchor_scene_pool_fingerprint": absent_anchor_pool[
                "scene_pool_fingerprint"
            ],
            "absent_anchor_support_index_fingerprint": (
                absent_support_index_fingerprint
            ),
            "absent_anchor_candidates_are_exact_raw_positive_captions": True,
            "absent_anchor_support_same_selection_partition": True,
            "absent_anchor_support_tier": "core",
            "absent_anchor_distinct_primary_support_scene_count": len(
                {row["primary_support_scene_id"] for row in absent_anchor_support}
            ),
            "qces_label_selection_used_for_absent_anchor_selection": False,
            "absent_anchor_must_be_confirmed_inaudible_by_both_raters": True,
        },
        "human_verification": {
            "status": "pending",
            "required_independent_passes": 2,
            "adjudication_required_on_disagreement": True,
            "required_region_fields": [
                "audible",
                "canonical_event_phrase",
                "verified_onset_seconds",
                "verified_offset_seconds",
                "contamination_rating",
                "salience_1_to_5",
            ],
            "required_scene_fields": [
                "event_inventory_complete_for_relations",
                "three_adjacent_onset_relations_verified",
                "verified_absent_anchor_label",
            ],
            "model_outputs_visible_to_annotators": False,
        },
    }


def verify_audio_archive(path: Path, filenames: Iterable[str]) -> Mapping[str, Any]:
    receipt = dict(_verify_official_file(path, "audio.zip"))
    selected = set(filenames)
    selected_members: dict[str, Mapping[str, Any]] = {}
    with zipfile.ZipFile(path, "r") as archive:
        seen: set[str] = set()
        unsafe_entries = 0
        symlink_entries = 0
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            unsafe = member.is_absolute() or ".." in member.parts
            unsafe_entries += int(unsafe)
            unix_mode = info.external_attr >> 16
            symlink_entries += int((unix_mode & 0o170000) == 0o120000)
            if unsafe:
                continue
            if len(member.parts) == 1 and member.suffix.casefold() == ".mp3":
                filename = member.name
                if filename in seen:
                    raise TacosAuditError(f"duplicate audio archive member: {filename}")
                seen.add(filename)
                if filename in selected:
                    selected_members[filename] = {
                        "archive_member": info.filename,
                        "uncompressed_size_bytes": info.file_size,
                        "compressed_size_bytes": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                    }
        if unsafe_entries or symlink_entries:
            raise TacosAuditError(
                f"unsafe audio archive: traversal={unsafe_entries}, symlinks={symlink_entries}"
            )
    missing = sorted(selected - set(selected_members))
    if missing:
        raise TacosAuditError(f"selected audio missing from archive: {missing[:5]}")
    receipt.update(
        {
            "archive_gate_passed": True,
            "unsafe_entries_↓": 0,
            "symlink_entries_↓": 0,
            "selected_members_verified_↑": len(selected_members),
            "selected_members": dict(sorted(selected_members.items())),
        }
    )
    return receipt


def build_tacos_real_plan(
    *,
    metadata_path: Path,
    strong_path: Path,
    weak_path: Path,
    development_split_path: Path,
    test_split_path: Path,
    record_json_path: Path,
    qces_label_selection_path: Path,
    seed: int,
    development_core_count: int,
    test_core_count: int,
    semantic_model_snapshot_path: Path,
    minimum_region_seconds: float = 0.25,
    minimum_onset_gap_seconds: float = 0.35,
    overlap_fraction: float = 0.5,
    benchmark_window_seconds: float = DEFAULT_BENCHMARK_WINDOW_SECONDS,
    maximum_pairwise_cosine: float = 0.80,
    cosine_round_decimals: int = 6,
    semantic_embedding_batch_size: int = 128,
    audio_archive_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Create a fixed-window, creator-disjoint, pre-human annotation packet."""

    if min(development_core_count, test_core_count) < 0:
        raise TacosAuditError("core counts cannot be negative")
    if development_core_count <= 0 or test_core_count <= 0:
        raise TacosAuditError("both core split counts must be positive")
    if benchmark_window_seconds != DEFAULT_BENCHMARK_WINDOW_SECONDS:
        raise TacosAuditError("the real benchmark window is frozen at exactly 10.0 s")
    if minimum_onset_gap_seconds <= 0.0:
        raise TacosAuditError("minimum onset gap must be positive")
    record = validate_record_json(record_json_path)
    label_selection = {
        **load_qces_label_selection(qces_label_selection_path),
        "legacy_provenance_only": True,
        "used_for_absent_anchor_selection": False,
    }
    input_receipts = {
        "metadata.csv": _verify_official_file(metadata_path, "metadata.csv"),
        "annotations_strong.csv": _verify_official_file(
            strong_path, "annotations_strong.csv"
        ),
        "annotations_weak.csv": _verify_official_file(
            weak_path, "annotations_weak.csv"
        ),
        "development_split.csv": _verify_official_file(
            development_split_path, "development_split.csv"
        ),
        "test_split.csv": _verify_official_file(test_split_path, "test_split.csv"),
    }
    candidates, _upstream_splits, diagnostics = _parse_inputs(
        metadata_path=metadata_path,
        strong_path=strong_path,
        weak_path=weak_path,
        development_split_path=development_split_path,
        test_split_path=test_split_path,
        minimum_region_seconds=minimum_region_seconds,
    )

    # Pool the official TACOS development and test partitions.  Their original
    # provenance remains attached to every scene, but the QCES split is assigned
    # at creator level by an immutable hash before either subset is selected.
    pre_embedding_pool = [
        candidate
        for candidate in candidates.values()
        if candidate.audio_license_spdx == CC0_SPDX
        and candidate.audio_license_url == CC0_URL
        and not candidate.source_metadata_reasons
        and candidate.usable_region_count >= 4
        and candidate.clip_duration >= benchmark_window_seconds
    ]
    semantic_captions = sorted(
        {
            normalize_caption(region.caption)
            for candidate in pre_embedding_pool
            for region in candidate.regions
            if region.eligible
        }
    )
    embeddings, semantic_model_receipt = encode_semantic_captions(
        semantic_captions,
        snapshot_path=semantic_model_snapshot_path,
        batch_size=semantic_embedding_batch_size,
    )
    ranked_candidates = [
        rank_candidate_chains(
            candidate,
            embeddings=embeddings,
            seed=seed,
            minimum_onset_gap_seconds=minimum_onset_gap_seconds,
            benchmark_window_seconds=benchmark_window_seconds,
            maximum_pairwise_cosine=maximum_pairwise_cosine,
            cosine_round_decimals=cosine_round_decimals,
        )
        for candidate in pre_embedding_pool
    ]
    structurally_feasible = [
        candidate for candidate in ranked_candidates if candidate.ranked_chain_count > 0
    ]
    semantic_pool = [
        candidate for candidate in ranked_candidates if candidate.best_chain is not None
    ]
    creator_assignments, creator_split_fingerprint = (
        assign_creator_hash_rank_partitions(
            candidate.creator_id for candidate in semantic_pool
        )
    )
    partitioned: dict[str, list[Candidate]] = {"real_dev": [], "real_test": []}
    for candidate in semantic_pool:
        partition = str(creator_assignments[candidate.creator_id]["partition"])
        partitioned[partition].append(candidate)

    partition_creator_counts = {
        key: len({candidate.creator_id for candidate in value})
        for key, value in partitioned.items()
    }
    capacity_summary = []
    capacity_failed = False
    for partition, needed in (
        ("real_dev", development_core_count),
        ("real_test", test_core_count),
    ):
        available = partition_creator_counts[partition]
        capacity_summary.append(
            f"{partition}: available={available}, required={needed}"
        )
        if available < needed:
            capacity_failed = True
    if capacity_failed:
        raise TacosAuditError(
            "post-semantic creator capacity is insufficient at maximum pairwise "
            f"cosine={maximum_pairwise_cosine:.6f}; "
            f"total_unique_creators={len({item.creator_id for item in semantic_pool})}; "
            + "; ".join(capacity_summary)
        )

    selected_development = select_balanced_candidates(
        partitioned["real_dev"],
        count=partition_creator_counts["real_dev"],
        seed=seed,
        split_name="real_dev",
        overlap_fraction=overlap_fraction,
    )
    selected_test = select_balanced_candidates(
        partitioned["real_test"],
        count=partition_creator_counts["real_test"],
        seed=seed,
        split_name="real_test",
        overlap_fraction=overlap_fraction,
    )

    selected_filenames = {
        item.filename for item in selected_test + selected_development
    }
    audio_receipt = (
        verify_audio_archive(audio_archive_path, selected_filenames)
        if audio_archive_path is not None
        else {
            "archive_gate_passed": False,
            "reason": "audio archive not supplied; metadata selection only",
        }
    )

    packet: list[dict[str, Any]] = []
    groups = (
        (selected_development[:development_core_count], "real_dev", "core"),
        (selected_development[development_core_count:], "real_dev", "reserve"),
        (selected_test[:test_core_count], "real_test", "core"),
        (selected_test[test_core_count:], "real_test", "reserve"),
    )
    selected_scene_partitions = {
        **{f"tacos_{item.freesound_id}": "real_dev" for item in selected_development},
        **{f"tacos_{item.freesound_id}": "real_test" for item in selected_test},
    }
    selected_scene_tiers = {
        **{
            f"tacos_{item.freesound_id}": (
                "core" if index < development_core_count else "reserve"
            )
            for index, item in enumerate(selected_development)
        },
        **{
            f"tacos_{item.freesound_id}": (
                "core" if index < test_core_count else "reserve"
            )
            for index, item in enumerate(selected_test)
        },
    }
    absent_anchor_pools, absent_anchor_audit = build_cross_scene_absent_anchor_pools(
        selected_development + selected_test,
        scene_partitions=selected_scene_partitions,
        scene_tiers=selected_scene_tiers,
        seed=seed,
        candidates_per_scene=6,
    )
    for items, partition, tier in groups:
        for ordinal, item in enumerate(items):
            if item.best_chain is None:
                raise AssertionError("selected scene lost its semantic chain")
            scene_id = f"tacos_{item.freesound_id}"
            packet.append(
                _scene_packet(
                    item,
                    partition=partition,
                    tier=tier,
                    ordinal=ordinal,
                    absent_anchor_pool=absent_anchor_pools[scene_id],
                    absent_support_index_fingerprint=str(
                        absent_anchor_audit["support_index_fingerprint"]
                    ),
                    creator_partition_hash=str(
                        creator_assignments[item.creator_id]["sha256"]
                    ),
                    creator_partition_rank=int(
                        creator_assignments[item.creator_id]["hash_rank"]
                    ),
                    maximum_pairwise_cosine=maximum_pairwise_cosine,
                    cosine_round_decimals=cosine_round_decimals,
                    benchmark_window_seconds=benchmark_window_seconds,
                )
            )

    selected_creator_ids = [row["source"]["creator_id"] for row in packet]
    if len(selected_creator_ids) != len(set(selected_creator_ids)):
        raise TacosAuditError("selected packet reuses a creator")
    dev_ids = {
        row["source"]["freesound_id"]
        for row in packet
        if row["selection_partition"] == "real_dev"
    }
    test_ids = {
        row["source"]["freesound_id"]
        for row in packet
        if row["selection_partition"] == "real_test"
    }
    if dev_ids & test_ids:
        raise TacosAuditError("selected real dev/test source overlap")
    dev_creators = {
        row["source"]["creator_id"]
        for row in packet
        if row["selection_partition"] == "real_dev"
    }
    test_creators = {
        row["source"]["creator_id"]
        for row in packet
        if row["selection_partition"] == "real_test"
    }
    if dev_creators & test_creators:
        raise TacosAuditError("selected real dev/test creator overlap")
    if any(len(row["suggested_distinct_onset_chain"]) != 4 for row in packet):
        raise TacosAuditError("selected packet contains a non-four-region chain")
    if any(
        row["benchmark_window"]["end_sample"]
        > row["benchmark_window"]["conservative_source_frame_budget"]
        for row in packet
    ):
        raise TacosAuditError(
            "selected packet exceeds conservative source frame budget"
        )
    positive_captions_by_scene = {
        row["scene_id"]: {
            region["caption"]
            for region in row["proposal_regions"]
            if region["selected_chain_member"]
        }
        for row in packet
    }
    packet_partition_by_scene = {
        row["scene_id"]: row["selection_partition"] for row in packet
    }
    packet_tier_by_scene = {row["scene_id"]: row["selection_tier"] for row in packet}
    unsupported_negative_captions = 0
    cross_partition_support_violations = 0
    noncore_support_violations = 0
    non_distinct_primary_support_sets = 0
    for row in packet:
        contract = row["question_contract"]
        candidates_in_packet = contract["absent_anchor_candidates"]
        support_rows = contract["absent_anchor_cross_scene_support"]
        unsupported_negative_captions += sum(
            support["caption"] != caption
            or support["cross_scene_support_count_↑"]
            != len(support["cross_scene_support_scene_ids"])
            or not support["cross_scene_support_scene_ids"]
            or row["scene_id"] in support["cross_scene_support_scene_ids"]
            or any(
                support["caption"]
                not in positive_captions_by_scene.get(support_scene_id, set())
                for support_scene_id in support["cross_scene_support_scene_ids"]
            )
            for caption, support in zip(candidates_in_packet, support_rows, strict=True)
        )
        cross_partition_support_violations += sum(
            support["selection_partition"] != row["selection_partition"]
            or any(
                packet_partition_by_scene.get(support_scene_id)
                != row["selection_partition"]
                for support_scene_id in support["cross_scene_support_scene_ids"]
            )
            for support in support_rows
        )
        noncore_support_violations += sum(
            support["support_tier"] != "core"
            or packet_partition_by_scene.get(support["primary_support_scene_id"])
            != row["selection_partition"]
            or packet_tier_by_scene.get(support["primary_support_scene_id"]) != "core"
            or any(
                packet_tier_by_scene.get(support_scene_id) != "core"
                for support_scene_id in support["cross_scene_support_scene_ids"]
            )
            for support in support_rows
        )
        non_distinct_primary_support_sets += int(
            len({support["primary_support_scene_id"] for support in support_rows})
            != len(support_rows)
        )
    if unsupported_negative_captions:
        raise TacosAuditError(
            "packet contains unsupported cross-scene negative captions"
        )
    if cross_partition_support_violations:
        raise TacosAuditError("packet contains cross-partition absent-anchor support")
    if noncore_support_violations or non_distinct_primary_support_sets:
        raise TacosAuditError("packet violates distinct core support-scene contract")

    packet_fingerprint = canonical_json_sha256(packet)
    input_fingerprint = canonical_json_sha256(
        {
            "record": record,
            "qces_label_selection": label_selection,
            "inputs": input_receipts,
            "semantic_model": semantic_model_receipt,
            "seed": seed,
            "counts": {
                "development_core": development_core_count,
                "test_core": test_core_count,
                "development_reserve": (
                    partition_creator_counts["real_dev"] - development_core_count
                ),
                "test_reserve": (
                    partition_creator_counts["real_test"] - test_core_count
                ),
            },
            "minimum_region_seconds": minimum_region_seconds,
            "minimum_onset_gap_seconds": minimum_onset_gap_seconds,
            "overlap_fraction": overlap_fraction,
            "benchmark_window_seconds": benchmark_window_seconds,
            "benchmark_sample_rate": BENCHMARK_SAMPLE_RATE,
            "source_duration_safety_seconds": SOURCE_DURATION_SAFETY_SECONDS,
            "source_duration_safety_samples": SOURCE_DURATION_SAFETY_SAMPLES,
            "maximum_pairwise_cosine": maximum_pairwise_cosine,
            "cosine_round_decimals": cosine_round_decimals,
            "creator_split_fingerprint": creator_split_fingerprint,
            "cross_scene_absent_anchor_pool": absent_anchor_audit,
            "semantic_embedding_batch_size": semantic_embedding_batch_size,
        }
    )
    selection_counts = Counter(
        f"{row['selection_partition']}:{row['selection_tier']}" for row in packet
    )
    core_rows = [row for row in packet if row["selection_tier"] == "core"]
    test_core_rows = [
        row for row in core_rows if row["selection_partition"] == "real_test"
    ]
    test_classes = {row["source"]["subclass"] for row in test_core_rows}
    test_overlap_scenes = sum(
        row["proposal_diagnostics"]["overlap_pair_count_↑"] > 0
        for row in test_core_rows
    )
    selected_upstream_counts = Counter(
        row["source"]["upstream_tacos_split"] for row in packet
    )
    selected_partition_upstream_counts = Counter(
        f"{row['selection_partition']}:{row['source']['upstream_tacos_split']}"
        for row in packet
    )
    structural_creator_count = len(
        {candidate.creator_id for candidate in structurally_feasible}
    )
    semantic_creator_count = len({candidate.creator_id for candidate in semantic_pool})
    selected_max_cosines = [
        float(row["semantic_chain"]["max_pairwise_cosine_↓"]) for row in packet
    ]
    selected_overlap_pairs = [
        int(row["proposal_diagnostics"]["overlap_pair_count_↑"]) for row in packet
    ]
    selected_concurrency = [
        int(row["proposal_diagnostics"]["max_concurrent_selected_regions_↑"])
        for row in packet
    ]
    event_center_histogram = [0] * 10
    event_middle_count = 0
    selected_event_count = 0
    start_quantile_histogram = [0] * 5
    variable_start_intervals = 0
    start_middle_count = 0
    for row in packet:
        for region in row["proposal_regions"]:
            if not region["selected_chain_member"]:
                continue
            center = (
                float(region["onset_seconds"]) + float(region["offset_seconds"])
            ) / 2
            event_center_histogram[min(9, max(0, int(center)))] += 1
            event_middle_count += int(4.0 <= center < 6.0)
            selected_event_count += 1
        low, high = row["benchmark_window"]["valid_start_sample_interval_inclusive"]
        if high > low:
            fraction = (row["benchmark_window"]["start_sample"] - low) / (high - low)
            start_quantile_histogram[min(4, int(fraction * 5))] += 1
            start_middle_count += int(0.4 <= fraction < 0.6)
            variable_start_intervals += 1
    event_middle_fraction = event_middle_count / selected_event_count
    start_middle_fraction = (
        start_middle_count / variable_start_intervals
        if variable_start_intervals
        else 0.0
    )
    compliance = {
        "format": COMPLIANCE_FORMAT,
        "input_fingerprint": input_fingerprint,
        "packet_fingerprint": packet_fingerprint,
        "metadata_selection_gate_passed": True,
        "audio_archive_gate_passed": bool(audio_receipt["archive_gate_passed"]),
        "human_verification_gate_passed": False,
        "submission_real_data_gate_passed": False,
        "gate_reason": (
            "The immutable CC0 candidate packet is ready, but two-pass human "
            "event verification, question finalization, and audio hashes remain pending."
        ),
        "metrics": {
            **diagnostics,
            "combined_cc0_pre_embedding_scenes_↑": len(pre_embedding_pool),
            "fixed_window_structurally_feasible_scenes_↑": len(structurally_feasible),
            "fixed_window_structurally_feasible_creators_↑": (structural_creator_count),
            "post_semantic_threshold_scenes_↑": len(semantic_pool),
            "post_semantic_threshold_creators_↑": semantic_creator_count,
            "post_hash_real_dev_scenes_↑": len(partitioned["real_dev"]),
            "post_hash_real_dev_creators_↑": partition_creator_counts["real_dev"],
            "post_hash_real_test_scenes_↑": len(partitioned["real_test"]),
            "post_hash_real_test_creators_↑": partition_creator_counts["real_test"],
            "selected_scenes_↑": len(packet),
            "selected_unique_sources_↑": len(selected_filenames),
            "selected_unique_creators_↑": len(set(selected_creator_ids)),
            "selected_unknown_audio_licenses_↓": 0,
            "selected_missing_attributions_↓": 0,
            "selected_dev_test_source_overlap_↓": 0,
            "selected_dev_test_creator_overlap_↓": 0,
            "selected_non_four_region_chains_↓": 0,
            "selected_non_10_second_windows_↓": 0,
            "selected_windows_exceeding_conservative_frame_budget_↓": 0,
            "absent_anchor_negative_candidate_instances_↑": (
                absent_anchor_audit["negative_candidate_instances_↑"]
            ),
            "absent_anchor_unique_negative_raw_captions_↑": (
                absent_anchor_audit["selected_negative_raw_caption_vocabulary_↑"]
            ),
            "absent_anchor_unsupported_negative_captions_↓": (
                unsupported_negative_captions
            ),
            "absent_anchor_cross_partition_support_violations_↓": (
                cross_partition_support_violations
            ),
            "absent_anchor_noncore_support_violations_↓": (noncore_support_violations),
            "absent_anchor_non_distinct_primary_support_sets_↓": (
                non_distinct_primary_support_sets
            ),
            "absent_anchor_distinct_primary_support_scenes_per_scene_↑": 6,
            "absent_anchor_negative_positive_support_fraction_↑": (
                absent_anchor_audit["negative_positive_vocabulary_support_fraction_↑"]
            ),
            "synthetic_qces_labels_used_for_absent_anchor_selection_↓": 0,
            "question_only_negative_vocabulary_shortcut_violations_↓": 0,
            "selected_max_pairwise_cosine_↓": max(selected_max_cosines),
            "selected_scenes_with_pairwise_overlap_↑": sum(
                count > 0 for count in selected_overlap_pairs
            ),
            "selected_pairwise_overlap_ratio_↑": (
                sum(selected_overlap_pairs) / (6 * len(selected_overlap_pairs))
            ),
            "selected_mean_overlap_pairs_per_scene_↑": (
                sum(selected_overlap_pairs) / len(selected_overlap_pairs)
            ),
            "selected_max_concurrency_histogram_1_to_4_↑": [
                selected_concurrency.count(level) for level in range(1, 5)
            ],
            "selected_scenes_with_four_concurrent_regions_↑": (
                selected_concurrency.count(4)
            ),
            "selected_event_center_position_histogram_0_to_10s_↑": (
                event_center_histogram
            ),
            "selected_empty_event_center_bins_↓": sum(
                count == 0 for count in event_center_histogram
            ),
            "selected_event_middle_20pct_abs_error_from_uniform_↓": abs(
                event_middle_fraction - 0.2
            ),
            "selected_variable_valid_start_intervals_↑": (variable_start_intervals),
            "selected_hash_start_quantile_histogram_↑": start_quantile_histogram,
            "selected_empty_hash_start_quantile_bins_↓": sum(
                count == 0 for count in start_quantile_histogram
            ),
            "selected_hash_start_middle_20pct_abs_error_from_uniform_↓": abs(
                start_middle_fraction - 0.2
            ),
            "selected_official_development_sources_↑": selected_upstream_counts[
                "development"
            ],
            "selected_official_test_sources_↑": selected_upstream_counts["test"],
            "real_dev_from_official_development_↑": (
                selected_partition_upstream_counts["real_dev:development"]
            ),
            "real_dev_from_official_test_↑": (
                selected_partition_upstream_counts["real_dev:test"]
            ),
            "real_test_from_official_development_↑": (
                selected_partition_upstream_counts["real_test:development"]
            ),
            "real_test_from_official_test_↑": (
                selected_partition_upstream_counts["real_test:test"]
            ),
            "real_dev_core_scenes_↑": selection_counts["real_dev:core"],
            "real_dev_reserve_scenes_↑": selection_counts["real_dev:reserve"],
            "real_test_core_scenes_↑": selection_counts["real_test:core"],
            "real_test_reserve_scenes_↑": selection_counts["real_test:reserve"],
            "real_test_core_question_capacity_↑": len(test_core_rows)
            * len(QUESTION_SLOTS),
            "real_test_core_subclasses_↑": len(test_classes),
            "real_test_core_overlap_scenes_↑": test_overlap_scenes,
            # Reserves must be annotated before model evaluation too; otherwise
            # accepting a reserve after seeing core failures would require a
            # second, adaptively chosen annotation pass.  Question rows are
            # generated only after the scene-level event/order audit, so do not
            # mislabel the 800-row capacity as 800 independent QA judgments.
            "pending_two_rater_scene_verifications_↓": len(packet),
            "pending_independent_scene_judgments_↓": 2 * len(packet),
            "provisional_core_scenes_↑": len(core_rows),
            "provisional_real_test_core_question_capacity_↑": (
                len(test_core_rows) * len(QUESTION_SLOTS)
            ),
        },
        "audio_archive": audio_receipt,
        "semantic_model": semantic_model_receipt,
        "cross_scene_absent_anchor_pool": absent_anchor_audit,
        "source_duration_safety": {
            "seconds": SOURCE_DURATION_SAFETY_SECONDS,
            "samples_at_32khz": SOURCE_DURATION_SAFETY_SAMPLES,
            "conservative_frame_budget_rule": (
                "floor(metadata_clip_duration_seconds * 32000) - 320"
            ),
        },
    }
    plan = {
        "format": PLAN_FORMAT,
        "dataset": DATASET_NAME,
        "record": record,
        "license_policy": {
            "primary_audio": "CC0-only",
            "allowed_audio_license_spdx": [CC0_SPDX],
            "caption_license_spdx": CAPTION_LICENSE_SPDX,
            "non_cc0_recordings_used": False,
        },
        "use_policy": {
            "training": False,
            "model_selection_on_real_test": False,
            "real_dev_may_be_used_for_protocol_debugging": True,
            "real_test_opened_once_after_method_freeze": True,
            "tacos_regions_are_proposals_not_clean_stems": True,
            "si_sdr_against_tacos_regions_prohibited": True,
            "real_test_audio_inspected_during_metadata_planning": False,
            "human_labels_used_for_window_or_chain_selection": False,
            "qces_method_outputs_used_for_window_or_chain_selection": False,
        },
        "selection": {
            "seed": seed,
            "minimum_region_seconds": minimum_region_seconds,
            "minimum_onset_gap_seconds": minimum_onset_gap_seconds,
            "target_overlap_fraction": overlap_fraction,
            "benchmark_window_seconds": benchmark_window_seconds,
            "benchmark_window_sample_rate_hz": BENCHMARK_SAMPLE_RATE,
            "source_duration_safety_seconds": SOURCE_DURATION_SAFETY_SECONDS,
            "source_duration_safety_samples": SOURCE_DURATION_SAFETY_SAMPLES,
            "conservative_source_frame_budget_rule": (
                "floor(metadata_clip_duration_seconds * 32000) - 320"
            ),
            "benchmark_window_start_rule": (
                "proposal-informed full-containment interval with sample-"
                "quantized hash-uniform start; frozen at benchmark construction, "
                "never runtime- or question-specific"
            ),
            "semantic_model_id": SEMANTIC_MODEL_ID,
            "semantic_model_revision": SEMANTIC_MODEL_REVISION,
            "semantic_model_weights_sha256": SEMANTIC_MODEL_SHA256,
            "semantic_embedding_implementation": SEMANTIC_EMBEDDING_IMPLEMENTATION,
            "maximum_pairwise_cosine": maximum_pairwise_cosine,
            "cosine_round_decimals": cosine_round_decimals,
            "exact_semantic_chain_size": 4,
            "custom_creator_split": {
                "performed_before_scene_selection": True,
                "hash_domain_hex": CUSTOM_SPLIT_HASH_DOMAIN.hex(),
                "digest_bits": 256,
                "rule": (
                    "sort all semantically eligible unique creators by "
                    "domain-separated SHA-256; first floor(N/5) are real_dev, "
                    "all remaining creators are real_test"
                ),
                "development_numerator": CUSTOM_DEVELOPMENT_NUMERATOR,
                "development_denominator": CUSTOM_DEVELOPMENT_DENOMINATOR,
                "eligible_unique_creators": semantic_creator_count,
                "real_dev_creators": partition_creator_counts["real_dev"],
                "real_test_creators": partition_creator_counts["real_test"],
                "split_fingerprint": creator_split_fingerprint,
                "all_eligible_creators_retained_one_scene": True,
                "official_development_and_test_combined": True,
                "upstream_split_provenance_preserved": True,
            },
            "one_scene_per_creator": True,
            "creator_disjoint_real_dev_test": True,
            "class_balancing": (
                "greedy least-represented subclass, overlap target, semantic "
                "score, stable hash tie-break"
            ),
            "counts": dict(sorted(selection_counts.items())),
            "position_prior_audit": {
                "event_center_histogram_bins_seconds": [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                ],
                "hash_start_quantile_bins": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "uniform_middle_bin_expected_fraction": 0.2,
                "purpose": "audit and disclose residual center-position prior",
            },
        },
        "question_contract": {
            "questions_per_accepted_scene": len(QUESTION_SLOTS),
            "slots": [dict(slot) for slot in QUESTION_SLOTS],
            "real_test_core_question_capacity": len(test_core_rows)
            * len(QUESTION_SLOTS),
            "absent_anchor_pool_source": "other_selected_chain_captions",
            "absent_anchor_candidates_per_scene": 6,
            "absent_anchor_support_index_fingerprint": absent_anchor_audit[
                "support_index_fingerprint"
            ],
            "absent_anchor_pool_fingerprint": absent_anchor_audit["pool_fingerprint"],
            "exact_raw_positive_caption_reuse": True,
            "support_scene_must_share_selection_partition": True,
            "support_tier": "core",
            "distinct_primary_support_scenes_per_scene": 6,
            "qces_label_selection_used_for_absent_anchor_selection": False,
            "human_labels_used_for_absent_anchor_selection": False,
            "qces_method_outputs_used_for_absent_anchor_selection": False,
        },
        "human_protocol": {
            "independent_event_verification_passes": 2,
            "adjudicate_disagreements": True,
            "model_outputs_hidden_during_dataset_annotation": True,
            "minimum_accepted_real_test_scenes": 100,
            "minimum_verified_real_test_questions": 800,
            "separate_blinded_output_listening_raters": 3,
        },
        "official_inputs": input_receipts,
        "semantic_model": semantic_model_receipt,
        "qces_label_selection": label_selection,
        "input_fingerprint": input_fingerprint,
        "packet_fingerprint": packet_fingerprint,
        "paper_result_eligible": False,
    }
    return plan, compliance, packet
