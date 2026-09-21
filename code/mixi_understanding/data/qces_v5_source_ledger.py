"""Strict metadata gate for a releaseable FUSS/FSD50K QCES v5 ledger.

This module never downloads data.  It joins a normalized, pinned FUSS source
manifest to a normalized, pinned FSD50K label/creator manifest, rejects unsafe
rows, plans source-disjoint QCES partitions, and optionally verifies already
acquired local audio before emitting a finalized receipt.

The normalized manifest contract is intentionally smaller and stricter than
the upstream archives.  See ``QCES_V5_SOURCE_ACQUISITION.md`` for the exact
conversion contract and upstream citations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


PLAN_FORMAT = "qces_v5_fuss_fsd50k_source_plan_v1"
RECEIPT_FORMAT = "qces_v5_audited_source_ledger_v1"
COMPLIANCE_FORMAT = "qces_v5_source_compliance_v1"
PINS_FORMAT = "qces_v5_fuss_fsd50k_pins_v1"
SELECTION_FORMAT = "qces_v5_source_selection_v1"
SOURCE_ROUTE = "fuss_v1.3_fsd50k_labels"
PROFILE = "paper"

FUSS_DATASET = "FUSS"
FUSS_VERSION = "1.3"
FUSS_DOI = "10.5281/zenodo.4012661"
FSD50K_DATASET = "FSD50K"
FSD50K_VERSION = "1.0"
FSD50K_DOI = "10.5281/zenodo.4060432"

CC0_SPDX = "CC0-1.0"
CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_BY_SPDX = "CC-BY-4.0"
CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/"
FSD50K_UPLOADER_KEY_PREFIX = "fsd50k-uploader-sha256:"
FSD50K_UPLOADER_KEY_DOMAIN = b"qces-v5-fsd50k-uploader-v1\0"

TARGET_PARTITIONS = (
    "train",
    "val",
    "test_iid",
    "test_compositional_ood",
    "test_label_ood",
)
FUSS_SPLITS = ("train", "validation", "eval")
TARGET_TO_FUSS_SPLIT = {
    "train": "train",
    "val": "validation",
    "test_iid": "eval",
    "test_compositional_ood": "eval",
    "test_label_ood": "eval",
}
ROLE_ALLOCATIONS: Mapping[str, Mapping[str, int]] = {
    "semantic_seen": {
        "train": 4,
        "val": 2,
        "test_iid": 2,
        "test_compositional_ood": 2,
    },
    "semantic_heldout": {"test_label_ood": 4},
    "nuisance": {
        "train": 2,
        "val": 1,
        "test_iid": 1,
        "test_compositional_ood": 1,
        "test_label_ood": 1,
    },
}
LABEL_MINIMA = {"seen_labels": 30, "heldout_labels": 10, "nuisance_labels": 8}
ROLE_MINIMUM_DURATION_SECONDS = {
    "semantic_seen": 1.5,
    "semantic_heldout": 1.5,
    "nuisance": 5.8,
}

_FUSS_FIELDS = {
    "source_id",
    "split",
    "audio_path",
    "sha256",
    "duration_seconds",
    "source_license_spdx",
    "source_license_url",
    "dataset_version",
}
_FSD_FIELDS = {
    "source_id",
    "split",
    "labels",
    "creator_id",
    "uploader_id",
    "uploader_name",
    "attribution",
    "source_license_spdx",
    "source_license_url",
    "dataset_version",
}
_PINS_FIELDS = {"schema_version", "audit_date", "fuss", "fsd50k"}
_FUSS_PIN_FIELDS = {
    "dataset",
    "dataset_version",
    "doi",
    "manifest_path",
    "manifest_sha256",
    "license_path",
    "license_sha256",
    "dataset_license_spdx",
    "dataset_license_url",
    "recipe_repository",
    "recipe_revision",
}
_FSD_PIN_FIELDS = {
    "dataset",
    "dataset_version",
    "doi",
    "manifest_path",
    "manifest_sha256",
    "license_path",
    "license_sha256",
    "dataset_license_spdx",
    "dataset_license_url",
}
_SELECTION_FIELDS = {
    "schema_version",
    "seen_labels",
    "heldout_labels",
    "nuisance_labels",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class SourceAuditError(ValueError):
    """Raised when pinned inputs or the audit contract itself are invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
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


def atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exact_fields(payload: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SourceAuditError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceAuditError(f"{context} must be a non-empty string")
    if value != value.strip():
        raise SourceAuditError(f"{context} must not have surrounding whitespace")
    return value


def fsd50k_uploader_key(uploader_name: Any, context: str = "uploader_name") -> str:
    """Derive the documented stable identity key from an exact username.

    FSD50K v1.0 publishes the Freesound uploader username, but not a numeric
    uploader identifier or a distinct creator identifier.  The route therefore
    domain-separates and hashes the exact UTF-8 username.  It deliberately does
    not case-fold or apply Unicode normalization, and the result must never be
    represented as an upstream numeric identifier.
    """

    username = _nonempty(uploader_name, context)
    digest = hashlib.sha256(
        FSD50K_UPLOADER_KEY_DOMAIN + username.encode("utf-8")
    ).hexdigest()
    return FSD50K_UPLOADER_KEY_PREFIX + digest


def _sha256(value: Any, context: str) -> str:
    result = _nonempty(value, context)
    if not _SHA256_RE.fullmatch(result):
        raise SourceAuditError(f"{context} must be a lowercase SHA256 digest")
    return result


def _revision(value: Any, context: str) -> str:
    result = _nonempty(value, context)
    if not _REVISION_RE.fullmatch(result):
        raise SourceAuditError(f"{context} must be a pinned 40-hex revision")
    return result


def _positive_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceAuditError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SourceAuditError(f"{context} must be finite and positive")
    return result


def _safe_relative(value: Any, context: str) -> str:
    result = _nonempty(value, context)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or result != path.as_posix():
        raise SourceAuditError(f"{context} must be a normalized safe relative path")
    return result


def _resolve_under(root: Path, relative: str, context: str) -> Path:
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise SourceAuditError(f"{context} escapes its declared root") from error
    return resolved


def _project_relative(path: Path, project_root: Path, context: str) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise SourceAuditError(
            f"{context} must resolve inside project root {project_root.resolve()}"
        ) from error


def _normalize_license(value: Any, context: str) -> Tuple[str, str]:
    raw = _nonempty(value, context)
    key = raw.strip().lower().replace("_", "-")
    cc0 = {
        "cc0",
        "cc0-1.0",
        "creative commons 0",
        "creative commons zero",
        CC0_URL.lower(),
        "http://creativecommons.org/publicdomain/zero/1.0/",
    }
    cc_by = {
        "cc-by-4.0",
        "creative commons attribution 4.0",
        CC_BY_URL.lower(),
        "http://creativecommons.org/licenses/by/4.0/",
    }
    if key in cc0:
        return CC0_SPDX, CC0_URL
    if key in cc_by:
        return CC_BY_SPDX, CC_BY_URL
    return raw, ""


def _validate_license_pair(
    spdx: Any, url: Any, *, expected: str, context: str
) -> Tuple[str, str]:
    normalized_spdx, canonical_url = _normalize_license(spdx, f"{context}.spdx")
    normalized_url_spdx, _ = _normalize_license(url, f"{context}.url")
    if normalized_spdx != expected or normalized_url_spdx != expected:
        raise SourceAuditError(
            f"{context} is not the required {expected} license: "
            f"spdx={spdx!r}, url={url!r}"
        )
    return normalized_spdx, canonical_url


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceAuditError(f"cannot read {context} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SourceAuditError(f"{context} must be a JSON object")
    return payload


def _read_jsonl(path: Path, context: str) -> List[Tuple[int, Mapping[str, Any]]]:
    rows: List[Tuple[int, Mapping[str, Any]]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise SourceAuditError(
                        f"{context}:{line_number} contains a blank line"
                    )
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise SourceAuditError(
                        f"{context}:{line_number} is invalid JSON: {error}"
                    ) from error
                if not isinstance(payload, dict):
                    raise SourceAuditError(
                        f"{context}:{line_number} must be a JSON object"
                    )
                rows.append((line_number, payload))
    except OSError as error:
        raise SourceAuditError(f"cannot read {context} {path}: {error}") from error
    if not rows:
        raise SourceAuditError(f"{context} is empty: {path}")
    return rows


@dataclass(frozen=True)
class FUSSSource:
    source_id: str
    split: str
    audio_path: str
    sha256: str
    duration_seconds: float
    source_license_spdx: str
    source_license_url: str
    dataset_version: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], context: str) -> "FUSSSource":
        _exact_fields(payload, _FUSS_FIELDS, context)
        split = _nonempty(payload["split"], f"{context}.split")
        if split not in FUSS_SPLITS:
            raise SourceAuditError(f"{context}.split must be one of {FUSS_SPLITS}")
        version = _nonempty(payload["dataset_version"], f"{context}.dataset_version")
        if version != FUSS_VERSION:
            raise SourceAuditError(
                f"{context}.dataset_version must be pinned to {FUSS_VERSION}"
            )
        spdx, url = _validate_license_pair(
            payload["source_license_spdx"],
            payload["source_license_url"],
            expected=CC0_SPDX,
            context=f"{context}.source_license",
        )
        return cls(
            source_id=_nonempty(payload["source_id"], f"{context}.source_id"),
            split=split,
            audio_path=_safe_relative(payload["audio_path"], f"{context}.audio_path"),
            sha256=_sha256(payload["sha256"], f"{context}.sha256"),
            duration_seconds=_positive_number(
                payload["duration_seconds"], f"{context}.duration_seconds"
            ),
            source_license_spdx=spdx,
            source_license_url=url,
            dataset_version=version,
        )


@dataclass(frozen=True)
class FSD50KSource:
    source_id: str
    split: str
    labels: Tuple[str, ...]
    creator_id: str
    uploader_id: str
    uploader_name: str
    attribution: str
    source_license_spdx: str
    source_license_url: str
    dataset_version: str

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], context: str
    ) -> "FSD50KSource":
        _exact_fields(payload, _FSD_FIELDS, context)
        split = _nonempty(payload["split"], f"{context}.split")
        if split not in FUSS_SPLITS:
            raise SourceAuditError(f"{context}.split must be one of {FUSS_SPLITS}")
        labels_raw = payload["labels"]
        if not isinstance(labels_raw, list) or not labels_raw:
            raise SourceAuditError(f"{context}.labels must be a non-empty list")
        labels = tuple(
            _nonempty(value, f"{context}.labels[{index}]")
            for index, value in enumerate(labels_raw)
        )
        if len(set(labels)) != len(labels):
            raise SourceAuditError(f"{context}.labels contains duplicates")
        version = _nonempty(payload["dataset_version"], f"{context}.dataset_version")
        if version != FSD50K_VERSION:
            raise SourceAuditError(
                f"{context}.dataset_version must be pinned to {FSD50K_VERSION}"
            )
        spdx, url = _validate_license_pair(
            payload["source_license_spdx"],
            payload["source_license_url"],
            expected=CC0_SPDX,
            context=f"{context}.source_license",
        )
        uploader_name = _nonempty(
            payload["uploader_name"], f"{context}.uploader_name"
        )
        expected_identity = fsd50k_uploader_key(
            uploader_name, f"{context}.uploader_name"
        )
        creator_id = _nonempty(payload["creator_id"], f"{context}.creator_id")
        uploader_id = _nonempty(payload["uploader_id"], f"{context}.uploader_id")
        if creator_id != expected_identity or uploader_id != expected_identity:
            raise SourceAuditError(
                f"{context} creator_id and uploader_id must both equal the "
                "documented deterministic key derived from uploader_name"
            )
        return cls(
            source_id=_nonempty(payload["source_id"], f"{context}.source_id"),
            split=split,
            labels=labels,
            creator_id=creator_id,
            uploader_id=uploader_id,
            uploader_name=uploader_name,
            attribution=_nonempty(payload["attribution"], f"{context}.attribution"),
            source_license_spdx=spdx,
            source_license_url=url,
            dataset_version=version,
        )


@dataclass(frozen=True)
class JoinedSource:
    source_id: str
    label: str
    split: str
    audio_path: str
    audio_abspath: Path
    sha256: str
    duration_seconds: float
    creator_id: str
    uploader_id: str
    attribution: str
    source_license_spdx: str
    source_license_url: str


def _issue(
    *, source_id: str, stage: str, reason: str, line_number: int | None = None
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source_id": source_id,
        "stage": stage,
        "reason": reason,
    }
    if line_number is not None:
        payload["line_number"] = line_number
    return payload


def _load_manifest(
    path: Path, *, kind: str
) -> Tuple[Dict[str, FUSSSource | FSD50KSource], List[Dict[str, Any]], int]:
    parser = FUSSSource.from_dict if kind == "fuss" else FSD50KSource.from_dict
    parsed: List[Tuple[int, FUSSSource | FSD50KSource]] = []
    issues: List[Dict[str, Any]] = []
    raw_rows = _read_jsonl(path, f"{kind} manifest")
    for line_number, payload in raw_rows:
        source_hint = payload.get("source_id")
        if not isinstance(source_hint, str) or not source_hint.strip():
            source_hint = f"<line:{line_number}>"
        try:
            row = parser(payload, f"{kind}[{line_number}]")
        except SourceAuditError as error:
            issues.append(
                _issue(
                    source_id=str(source_hint),
                    stage=f"{kind}_manifest",
                    reason=str(error),
                    line_number=line_number,
                )
            )
            continue
        parsed.append((line_number, row))

    counts = Counter(row.source_id for _, row in parsed)
    duplicate_ids = {source_id for source_id, count in counts.items() if count > 1}
    result: Dict[str, FUSSSource | FSD50KSource] = {}
    for line_number, row in parsed:
        if row.source_id in duplicate_ids:
            issues.append(
                _issue(
                    source_id=row.source_id,
                    stage=f"{kind}_manifest",
                    reason="duplicate_source_id",
                    line_number=line_number,
                )
            )
            continue
        result[row.source_id] = row
    return result, sorted(issues, key=_issue_sort_key), len(raw_rows)


def _issue_sort_key(payload: Mapping[str, Any]) -> Tuple[str, str, str, int]:
    return (
        str(payload.get("source_id", "")),
        str(payload.get("stage", "")),
        str(payload.get("reason", "")),
        int(payload.get("line_number", 0)),
    )


def _validate_license_text(path: Path, context: str) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError as error:
        raise SourceAuditError(f"cannot read {context}: {path}: {error}") from error
    markers = (
        "creative commons attribution 4.0",
        "creativecommons.org/licenses/by/4.0",
        "cc by 4.0",
    )
    if not any(marker in text for marker in markers):
        raise SourceAuditError(
            f"{context} does not contain a recognizable CC BY 4.0 marker"
        )


def _validate_pin_block(
    payload: Mapping[str, Any], *, kind: str, root: Path
) -> Dict[str, Any]:
    expected_fields = _FUSS_PIN_FIELDS if kind == "fuss" else _FSD_PIN_FIELDS
    _exact_fields(payload, expected_fields, f"pins.{kind}")
    expected = (
        (FUSS_DATASET, FUSS_VERSION, FUSS_DOI)
        if kind == "fuss"
        else (FSD50K_DATASET, FSD50K_VERSION, FSD50K_DOI)
    )
    observed = (
        _nonempty(payload["dataset"], f"pins.{kind}.dataset"),
        _nonempty(payload["dataset_version"], f"pins.{kind}.dataset_version"),
        _nonempty(payload["doi"], f"pins.{kind}.doi"),
    )
    if observed != expected:
        raise SourceAuditError(
            f"pins.{kind} dataset/version/DOI mismatch: "
            f"expected={expected}, observed={observed}"
        )
    _validate_license_pair(
        payload["dataset_license_spdx"],
        payload["dataset_license_url"],
        expected=CC_BY_SPDX,
        context=f"pins.{kind}.dataset_license",
    )
    manifest_relative = _safe_relative(
        payload["manifest_path"], f"pins.{kind}.manifest_path"
    )
    license_relative = _safe_relative(
        payload["license_path"], f"pins.{kind}.license_path"
    )
    manifest_path = _resolve_under(root, manifest_relative, f"pins.{kind}.manifest")
    license_path = _resolve_under(root, license_relative, f"pins.{kind}.license")
    if not manifest_path.is_file():
        raise SourceAuditError(f"pinned {kind} manifest is missing: {manifest_path}")
    if not license_path.is_file():
        raise SourceAuditError(f"pinned {kind} license file is missing: {license_path}")
    expected_manifest_hash = _sha256(
        payload["manifest_sha256"], f"pins.{kind}.manifest_sha256"
    )
    expected_license_hash = _sha256(
        payload["license_sha256"], f"pins.{kind}.license_sha256"
    )
    actual_manifest_hash = sha256_file(manifest_path)
    actual_license_hash = sha256_file(license_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise SourceAuditError(
            f"{kind} manifest hash mismatch: expected={expected_manifest_hash}, "
            f"actual={actual_manifest_hash}"
        )
    if actual_license_hash != expected_license_hash:
        raise SourceAuditError(
            f"{kind} license hash mismatch: expected={expected_license_hash}, "
            f"actual={actual_license_hash}"
        )
    _validate_license_text(license_path, f"{kind} license file")
    result = dict(payload)
    result["manifest_abspath"] = manifest_path
    result["license_abspath"] = license_path
    if kind == "fuss":
        repository = _nonempty(
            payload["recipe_repository"], "pins.fuss.recipe_repository"
        )
        if repository != "https://github.com/google-research/sound-separation":
            raise SourceAuditError("pins.fuss.recipe_repository is not official")
        _revision(payload["recipe_revision"], "pins.fuss.recipe_revision")
    return result


def load_pins(
    pins_path: Path, *, fuss_root: Path, fsd50k_root: Path
) -> Dict[str, Any]:
    payload = _read_json(pins_path, "pins JSON")
    _exact_fields(payload, _PINS_FIELDS, "pins")
    if payload["schema_version"] != PINS_FORMAT:
        raise SourceAuditError(f"pins.schema_version must be {PINS_FORMAT}")
    audit_date = _nonempty(payload["audit_date"], "pins.audit_date")
    try:
        date.fromisoformat(audit_date)
    except ValueError as error:
        raise SourceAuditError("pins.audit_date must be ISO YYYY-MM-DD") from error
    if not isinstance(payload["fuss"], dict) or not isinstance(payload["fsd50k"], dict):
        raise SourceAuditError("pins.fuss and pins.fsd50k must be objects")
    return {
        "schema_version": PINS_FORMAT,
        "audit_date": audit_date,
        "pins_sha256": sha256_file(pins_path),
        "fuss": _validate_pin_block(payload["fuss"], kind="fuss", root=fuss_root),
        "fsd50k": _validate_pin_block(
            payload["fsd50k"], kind="fsd50k", root=fsd50k_root
        ),
    }


def load_selection(path: Path) -> Dict[str, List[str]]:
    payload = _read_json(path, "selection JSON")
    _exact_fields(payload, _SELECTION_FIELDS, "selection")
    if payload["schema_version"] != SELECTION_FORMAT:
        raise SourceAuditError(
            f"selection.schema_version must be {SELECTION_FORMAT}"
        )
    result: Dict[str, List[str]] = {}
    for key, minimum in LABEL_MINIMA.items():
        values = payload[key]
        if not isinstance(values, list):
            raise SourceAuditError(f"selection.{key} must be a list")
        labels = [_nonempty(value, f"selection.{key}[{i}]") for i, value in enumerate(values)]
        if len(set(labels)) != len(labels):
            raise SourceAuditError(f"selection.{key} contains duplicates")
        if len(labels) < minimum:
            raise SourceAuditError(
                f"selection.{key} has {len(labels)} labels; paper minimum is {minimum}"
            )
        result[key] = sorted(labels)
    sets = {key: set(values) for key, values in result.items()}
    for left, right in (
        ("seen_labels", "heldout_labels"),
        ("seen_labels", "nuisance_labels"),
        ("heldout_labels", "nuisance_labels"),
    ):
        overlap = sorted(sets[left] & sets[right])
        if overlap:
            raise SourceAuditError(
                f"selection label roles overlap ({left}, {right}): {overlap}"
            )
    return result


def _join_sources(
    fuss: Mapping[str, FUSSSource | FSD50KSource],
    fsd: Mapping[str, FUSSSource | FSD50KSource],
    *,
    fuss_root: Path,
    project_root: Path,
) -> Tuple[List[JoinedSource], List[Dict[str, Any]]]:
    joined: List[JoinedSource] = []
    rejections: List[Dict[str, Any]] = []
    for source_id in sorted(fuss):
        fuss_row = fuss[source_id]
        if not isinstance(fuss_row, FUSSSource):
            raise AssertionError("internal FUSS row type mismatch")
        fsd_row = fsd.get(source_id)
        if not isinstance(fsd_row, FSD50KSource):
            rejections.append(
                _issue(
                    source_id=source_id,
                    stage="join",
                    reason="unmatched_fsd50k_source_id",
                )
            )
            continue
        if fuss_row.split != fsd_row.split:
            rejections.append(
                _issue(
                    source_id=source_id,
                    stage="join",
                    reason=(
                        "split_mismatch:"
                        f"fuss={fuss_row.split},fsd50k={fsd_row.split}"
                    ),
                )
            )
            continue
        if len(fsd_row.labels) != 1:
            rejections.append(
                _issue(
                    source_id=source_id,
                    stage="join",
                    reason="multilabel_ambiguous:" + "|".join(sorted(fsd_row.labels)),
                )
            )
            continue
        if (
            fuss_row.source_license_spdx != CC0_SPDX
            or fsd_row.source_license_spdx != CC0_SPDX
        ):
            rejections.append(
                _issue(
                    source_id=source_id,
                    stage="join",
                    reason="source_license_not_cc0",
                )
            )
            continue
        audio_abspath = _resolve_under(
            fuss_root, fuss_row.audio_path, f"FUSS source {source_id} audio_path"
        )
        try:
            project_audio_path = _project_relative(
                audio_abspath, project_root, f"FUSS source {source_id} audio_path"
            )
        except SourceAuditError as error:
            rejections.append(
                _issue(
                    source_id=source_id,
                    stage="join",
                    reason=str(error),
                )
            )
            continue
        joined.append(
            JoinedSource(
                source_id=source_id,
                label=fsd_row.labels[0],
                split=fuss_row.split,
                audio_path=project_audio_path,
                audio_abspath=audio_abspath,
                sha256=fuss_row.sha256,
                duration_seconds=fuss_row.duration_seconds,
                creator_id=fsd_row.creator_id,
                uploader_id=fsd_row.uploader_id,
                attribution=fsd_row.attribution,
                source_license_spdx=CC0_SPDX,
                source_license_url=CC0_URL,
            )
        )

    for source_id in sorted(set(fsd) - set(fuss)):
        rejections.append(
            _issue(
                source_id=source_id,
                stage="join",
                reason="unmatched_fuss_source_id",
            )
        )

    hash_counts = Counter(row.sha256 for row in joined)
    duplicate_hashes = {digest for digest, count in hash_counts.items() if count > 1}
    deduplicated: List[JoinedSource] = []
    for row in joined:
        if row.sha256 in duplicate_hashes:
            rejections.append(
                _issue(
                    source_id=row.source_id,
                    stage="join",
                    reason=f"duplicate_content_sha256:{row.sha256}",
                )
            )
        else:
            deduplicated.append(row)
    return deduplicated, sorted(rejections, key=_issue_sort_key)


def _stable_key(seed: int, *parts: str) -> Tuple[str, ...]:
    value = "\0".join((str(seed),) + parts).encode("utf-8")
    return (hashlib.sha256(value).hexdigest(),) + parts


def _role_labels(selection: Mapping[str, Sequence[str]]) -> Dict[str, Sequence[str]]:
    return {
        "semantic_seen": selection["seen_labels"],
        "semantic_heldout": selection["heldout_labels"],
        "nuisance": selection["nuisance_labels"],
    }


def _select_sources(
    joined: Sequence[JoinedSource],
    *,
    selection: Mapping[str, Sequence[str]],
    seed: int,
    license_record_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_label_split: MutableMapping[Tuple[str, str], List[JoinedSource]] = defaultdict(list)
    for row in joined:
        by_label_split[(row.label, row.split)].append(row)
    for key, values in by_label_split.items():
        values.sort(key=lambda row: _stable_key(seed, key[0], key[1], row.source_id))

    demands: List[Dict[str, Any]] = []
    for role, labels in _role_labels(selection).items():
        minimum_duration = ROLE_MINIMUM_DURATION_SECONDS[role]
        for label in labels:
            for partition, count in ROLE_ALLOCATIONS[role].items():
                upstream_split = TARGET_TO_FUSS_SPLIT[partition]
                candidates = [
                    row
                    for row in by_label_split.get((label, upstream_split), ())
                    if row.duration_seconds + 1e-9 >= minimum_duration
                ]
                demands.append(
                    {
                        "role": role,
                        "label": label,
                        "partition": partition,
                        "upstream_split": upstream_split,
                        "count": count,
                        "candidates": candidates,
                        "candidate_uploader_count": len(
                            {row.uploader_id for row in candidates}
                        ),
                    }
                )
    # Scarce label/split demands reserve uploaders first.  Ties are explicit,
    # so changing JSONL order cannot change the plan.
    partition_rank = {partition: index for index, partition in enumerate(TARGET_PARTITIONS)}
    demands.sort(
        key=lambda demand: (
            demand["candidate_uploader_count"] - demand["count"],
            len(demand["candidates"]) - demand["count"],
            partition_rank[demand["partition"]],
            demand["role"],
            demand["label"],
        )
    )

    used_source_ids: set[str] = set()
    used_hashes: set[str] = set()
    creator_partition: Dict[str, str] = {}
    uploader_partition: Dict[str, str] = {}
    selected: List[Dict[str, Any]] = []
    shortfalls: List[Dict[str, Any]] = []
    for demand in demands:
        chosen: List[JoinedSource] = []
        candidates = sorted(
            demand["candidates"],
            key=lambda row: _stable_key(
                seed,
                demand["partition"],
                demand["role"],
                demand["label"],
                row.source_id,
            ),
        )
        for row in candidates:
            if row.source_id in used_source_ids or row.sha256 in used_hashes:
                continue
            creator_owner = creator_partition.get(row.creator_id)
            uploader_owner = uploader_partition.get(row.uploader_id)
            if creator_owner not in (None, demand["partition"]):
                continue
            if uploader_owner not in (None, demand["partition"]):
                continue
            chosen.append(row)
            used_source_ids.add(row.source_id)
            used_hashes.add(row.sha256)
            creator_partition[row.creator_id] = demand["partition"]
            uploader_partition[row.uploader_id] = demand["partition"]
            if len(chosen) == demand["count"]:
                break
        if len(chosen) != demand["count"]:
            shortfalls.append(
                {
                    "role": demand["role"],
                    "label": demand["label"],
                    "partition": demand["partition"],
                    "required": demand["count"],
                    "selected": len(chosen),
                    "missing": demand["count"] - len(chosen),
                    "eligible_before_isolation": len(candidates),
                    "eligible_uploaders_before_isolation": demand[
                        "candidate_uploader_count"
                    ],
                }
            )
        for row in chosen:
            selected.append(
                {
                    "source_id": row.source_id,
                    "label": row.label,
                    "role": demand["role"],
                    "partition": demand["partition"],
                    "source_dataset": FUSS_DATASET,
                    "dataset_version": FUSS_VERSION,
                    "creator_id": row.creator_id,
                    "uploader_id": row.uploader_id,
                    "attribution": row.attribution,
                    "source_license_spdx": row.source_license_spdx,
                    "source_license_url": row.source_license_url,
                    "source_interval_seconds": [0.0, row.duration_seconds],
                    "interval_duration_seconds": row.duration_seconds,
                    "audio_path": row.audio_path,
                    "acquired": False,
                    "sha256": row.sha256,
                    "license_record_id": license_record_id,
                }
            )
    return (
        sorted(selected, key=lambda row: row["source_id"]),
        sorted(
            shortfalls,
            key=lambda row: (row["partition"], row["role"], row["label"]),
        ),
    )


def _leakage_count(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    partitions: MutableMapping[str, set[str]] = defaultdict(set)
    for row in rows:
        partitions[str(row[field])].add(str(row["partition"]))
    return sum(len(values) > 1 for values in partitions.values())


def _metric(value: int | float, direction: str) -> Dict[str, Any]:
    if direction not in {"↑", "↓"}:
        raise AssertionError("metric direction must be an arrow")
    return {"value": value, "direction": direction}


def _license_record(pins: Mapping[str, Any]) -> Dict[str, Any]:
    evidence_fingerprint = canonical_json_sha256(
        {
            "pins_sha256": pins["pins_sha256"],
            "fuss_manifest_sha256": pins["fuss"]["manifest_sha256"],
            "fuss_license_sha256": pins["fuss"]["license_sha256"],
            "fsd50k_manifest_sha256": pins["fsd50k"]["manifest_sha256"],
            "fsd50k_license_sha256": pins["fsd50k"]["license_sha256"],
        }
    )
    return {
        "record_id": f"qces-v5-fuss-fsd50k-{evidence_fingerprint[:16]}",
        "status": "verified",
        "declared_license": CC_BY_SPDX,
        "redistribution_allowed": True,
        "commercial_use_allowed": True,
        "derivatives_allowed": True,
        "checked_on": pins["audit_date"],
        "evidence_urls": [
            f"https://doi.org/{FUSS_DOI}",
            f"https://doi.org/{FSD50K_DOI}",
            "https://github.com/google-research/sound-separation/tree/master/datasets/fuss",
        ],
        "evidence_fingerprint": evidence_fingerprint,
        "scope": (
            "FUSS v1.3 CC0 source audio; FUSS/FSD50K curated metadata and "
            "the derived QCES collection under CC-BY-4.0"
        ),
    }


def _selection_payload(selection: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    return {
        "seen_labels": list(selection["seen_labels"]),
        "heldout_labels": list(selection["heldout_labels"]),
        "nuisance_labels": list(selection["nuisance_labels"]),
        "seen_allocations": dict(ROLE_ALLOCATIONS["semantic_seen"]),
        "heldout_allocations": dict(ROLE_ALLOCATIONS["semantic_heldout"]),
        "nuisance_allocations": dict(ROLE_ALLOCATIONS["nuisance"]),
        "semantic_minimum_interval_seconds": ROLE_MINIMUM_DURATION_SECONDS[
            "semantic_seen"
        ],
        "nuisance_minimum_interval_seconds": ROLE_MINIMUM_DURATION_SECONDS[
            "nuisance"
        ],
    }


def _artifact_payload(pins: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "dataset": FUSS_DATASET,
            "dataset_version": FUSS_VERSION,
            "doi": FUSS_DOI,
            "manifest_path": pins["fuss"]["manifest_path"],
            "manifest_sha256": pins["fuss"]["manifest_sha256"],
            "license_path": pins["fuss"]["license_path"],
            "license_sha256": pins["fuss"]["license_sha256"],
            "dataset_license_spdx": CC_BY_SPDX,
            "recipe_repository": pins["fuss"]["recipe_repository"],
            "recipe_revision": pins["fuss"]["recipe_revision"],
        },
        {
            "dataset": FSD50K_DATASET,
            "dataset_version": FSD50K_VERSION,
            "doi": FSD50K_DOI,
            "manifest_path": pins["fsd50k"]["manifest_path"],
            "manifest_sha256": pins["fsd50k"]["manifest_sha256"],
            "license_path": pins["fsd50k"]["license_path"],
            "license_sha256": pins["fsd50k"]["license_sha256"],
            "dataset_license_spdx": CC_BY_SPDX,
            "usage_scope": "labels, creator/uploader provenance, and license metadata only",
        },
    ]


def _reason_count(issues: Iterable[Mapping[str, Any]], fragment: str) -> int:
    return sum(fragment in str(issue.get("reason", "")) for issue in issues)


def audit_source_ledger(
    *,
    fuss_root: Path,
    fsd50k_root: Path,
    pins_path: Path,
    selection_path: Path,
    project_root: Path,
    seed: int = 314_159,
    verify_audio: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any] | None]:
    """Audit and plan sources; finalize a receipt only after audio verification.

    Returns ``(plan, compliance, receipt_or_none)``.  Pinned-input corruption
    raises :class:`SourceAuditError`; row-level defects are reported and
    rejected, and only fail the metadata gate if the paper allocation cannot be
    completed from the remaining clean pool.
    """

    fuss_root = fuss_root.resolve()
    fsd50k_root = fsd50k_root.resolve()
    project_root = project_root.resolve()
    pins_path = pins_path.resolve()
    selection_path = selection_path.resolve()
    pins_project_path = _project_relative(
        pins_path, project_root, "pins JSON"
    )
    selection_project_path = _project_relative(
        selection_path, project_root, "selection JSON"
    )
    pins = load_pins(pins_path, fuss_root=fuss_root, fsd50k_root=fsd50k_root)
    selection = load_selection(selection_path)
    fuss_rows, fuss_issues, fuss_raw_count = _load_manifest(
        pins["fuss"]["manifest_abspath"], kind="fuss"
    )
    fsd_rows, fsd_issues, fsd_raw_count = _load_manifest(
        pins["fsd50k"]["manifest_abspath"], kind="fsd50k"
    )
    joined, join_rejections = _join_sources(
        fuss_rows,
        fsd_rows,
        fuss_root=fuss_root,
        project_root=project_root,
    )
    all_issues = sorted(
        fuss_issues + fsd_issues + join_rejections, key=_issue_sort_key
    )
    license_record = _license_record(pins)
    selected, shortfalls = _select_sources(
        joined,
        selection=selection,
        seed=seed,
        license_record_id=license_record["record_id"],
    )
    source_leakage = _leakage_count(selected, "source_id")
    creator_leakage = _leakage_count(selected, "creator_id")
    uploader_leakage = _leakage_count(selected, "uploader_id")
    hash_leakage = _leakage_count(selected, "sha256")
    metadata_gate_passed = (
        not shortfalls
        and source_leakage == 0
        and creator_leakage == 0
        and uploader_leakage == 0
        and hash_leakage == 0
    )

    counts_by_partition = Counter(row["partition"] for row in selected)
    plan: Dict[str, Any] = {
        "format": PLAN_FORMAT,
        "profile": PROFILE,
        "seed": seed,
        "source_route": SOURCE_ROUTE,
        "dataset": "FUSS+FSD50K-labels",
        "dataset_revision": f"FUSS-{FUSS_VERSION}+FSD50K-{FSD50K_VERSION}",
        # Downstream validators require one real project-local metadata anchor.
        # The pins JSON is that anchor and transitively commits both normalized
        # manifests, both license files, and the FUSS recipe revision.
        "metadata_path": pins_project_path,
        "metadata_sha256": pins["pins_sha256"],
        "pins_sha256": pins["pins_sha256"],
        "selection_path": selection_project_path,
        "selection_sha256": sha256_file(selection_path),
        "selection": _selection_payload(selection),
        "input_artifacts": _artifact_payload(pins),
        "source_count": len(selected),
        "source_counts_by_partition": {
            partition: counts_by_partition.get(partition, 0)
            for partition in TARGET_PARTITIONS
        },
        "acquisition_complete": False,
        "release_ready": False,
        "license_record": license_record,
        "sources": selected,
        "metadata_gate_passed": metadata_gate_passed,
        "non_claim": (
            "This plan proves only a pinned metadata join. It is not a finalized "
            "dataset receipt and does not claim that real audio is present."
        ),
    }

    acquisition_issues: List[Dict[str, Any]] = []
    verified_rows: List[Dict[str, Any]] = []
    if verify_audio and metadata_gate_passed:
        for row in selected:
            audio_path = (project_root / row["audio_path"]).resolve()
            if not audio_path.is_file():
                acquisition_issues.append(
                    _issue(
                        source_id=row["source_id"],
                        stage="audio_verification",
                        reason="missing_audio_file",
                    )
                )
                continue
            actual_hash = sha256_file(audio_path)
            if actual_hash != row["sha256"]:
                acquisition_issues.append(
                    _issue(
                        source_id=row["source_id"],
                        stage="audio_verification",
                        reason=(
                            f"audio_sha256_mismatch:expected={row['sha256']},"
                            f"actual={actual_hash}"
                        ),
                    )
                )
                continue
            committed = dict(row)
            committed["acquired"] = True
            verified_rows.append(committed)
    acquisition_gate_passed = bool(
        verify_audio
        and metadata_gate_passed
        and not acquisition_issues
        and len(verified_rows) == len(selected)
    )

    joined_denominator = max(1, len(fuss_rows))
    metrics = {
        "fuss_manifest_row_count": _metric(fuss_raw_count, "↑"),
        "fsd50k_manifest_row_count": _metric(fsd_raw_count, "↑"),
        "valid_joined_source_count": _metric(len(joined), "↑"),
        "valid_join_coverage": _metric(len(joined) / joined_denominator, "↑"),
        "admitted_source_count": _metric(len(selected), "↑"),
        "rejected_or_invalid_row_count": _metric(len(all_issues), "↓"),
        "unmatched_source_count": _metric(
            _reason_count(all_issues, "unmatched_fsd50k_source_id")
            + _reason_count(all_issues, "unmatched_fuss_source_id"),
            "↓",
        ),
        "multilabel_ambiguous_count": _metric(
            _reason_count(all_issues, "multilabel_ambiguous"), "↓"
        ),
        "missing_required_field_count": _metric(
            _reason_count(all_issues, "fields mismatch")
            + _reason_count(all_issues, "must be a non-empty string"),
            "↓",
        ),
        "non_cc0_source_count": _metric(
            _reason_count(all_issues, "required CC0-1.0")
            + _reason_count(all_issues, "source_license_not_cc0"),
            "↓",
        ),
        "duplicate_content_count": _metric(
            _reason_count(all_issues, "duplicate_content_sha256"), "↓"
        ),
        "allocation_shortfall_count": _metric(
            sum(row["missing"] for row in shortfalls), "↓"
        ),
        "source_split_leakage_count": _metric(source_leakage, "↓"),
        "creator_split_leakage_count": _metric(creator_leakage, "↓"),
        "uploader_split_leakage_count": _metric(uploader_leakage, "↓"),
        "content_hash_split_leakage_count": _metric(hash_leakage, "↓"),
        "verified_audio_count": _metric(len(verified_rows), "↑"),
        "audio_missing_or_hash_mismatch_count": _metric(
            len(acquisition_issues), "↓"
        ),
    }
    compliance: Dict[str, Any] = {
        "format": COMPLIANCE_FORMAT,
        "profile": PROFILE,
        "source_route": SOURCE_ROUTE,
        "status": (
            "fail"
            if not metadata_gate_passed
            else "pass"
            if not verify_audio or acquisition_gate_passed
            else "fail"
        ),
        "release_ready": acquisition_gate_passed,
        "gates": {
            "pinned_input_gate_passed": True,
            "dataset_license_gate_passed": True,
            "source_license_gate_passed": True,
            "metadata_join_gate_passed": metadata_gate_passed,
            "allocation_gate_passed": not shortfalls,
            "split_isolation_gate_passed": all(
                value == 0
                for value in (
                    source_leakage,
                    creator_leakage,
                    uploader_leakage,
                    hash_leakage,
                )
            ),
            "audio_verification_gate_run": verify_audio,
            "audio_verification_gate_passed": (
                acquisition_gate_passed if verify_audio else None
            ),
            "receipt_emitted": acquisition_gate_passed,
        },
        "metrics": metrics,
        "metric_directions": {
            key: value["direction"] for key, value in metrics.items()
        },
        "manifest_issues_and_rejections": all_issues,
        "allocation_shortfalls": shortfalls,
        "audio_verification_issues": sorted(
            acquisition_issues, key=_issue_sort_key
        ),
        "input_fingerprint": canonical_json_sha256(
            {
                "pins": pins["pins_sha256"],
                "selection": sha256_file(selection_path),
                "seed": seed,
            }
        ),
        "interpretation": {
            "↑": "higher is better",
            "↓": "lower is better",
            "plan_phase": (
                "A passing metadata gate is not release readiness; a receipt is "
                "emitted only after every selected local audio hash is verified."
            ),
        },
    }

    receipt: Dict[str, Any] | None = None
    if acquisition_gate_passed:
        receipt = dict(plan)
        receipt.update(
            {
                "format": RECEIPT_FORMAT,
                "acquisition_complete": True,
                "release_ready": True,
                "sources": sorted(verified_rows, key=lambda row: row["source_id"]),
                "metadata_gate_passed": True,
                "audio_verification": {
                    "verified_source_count": len(verified_rows),
                    "hash_algorithm": "SHA-256",
                    "all_declared_hashes_match": True,
                },
                "non_claim": (
                    "Finalized only for the locally verified, pinned artifacts "
                    "enumerated in this receipt; no broader upstream claim is made."
                ),
            }
        )
    return plan, compliance, receipt


__all__ = [
    "COMPLIANCE_FORMAT",
    "FUSS_DOI",
    "FSD50K_DOI",
    "PINS_FORMAT",
    "PLAN_FORMAT",
    "PROFILE",
    "RECEIPT_FORMAT",
    "ROLE_ALLOCATIONS",
    "SELECTION_FORMAT",
    "SOURCE_ROUTE",
    "SourceAuditError",
    "atomic_json",
    "audit_source_ledger",
    "canonical_json_sha256",
    "load_pins",
    "load_selection",
    "sha256_file",
]
