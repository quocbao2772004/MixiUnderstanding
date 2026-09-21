#!/usr/bin/env python3
"""Acquire an immutable, checksum-verified RealDESED split from Zenodo.

This command deliberately separates acquisition from QCES derivation.  It
stores the publisher metadata beside the untouched archive, verifies the
publisher checksum before extraction, rejects unsafe ZIP members, and emits a
machine-readable receipt.  It never generates questions or modifies source
audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ZENODO_RECORD_ID = "20056072"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
EXPECTED_FILES = {
    "train": {
        "name": "train.zip",
        "bytes": 5_647_895_421,
        "checksum": "md5:8d3f30f04949cd481a7fb14e2e73333a",
    },
    "validation": {
        "name": "validation.zip",
        "bytes": 1_586_397_692,
        "checksum": "md5:5170442e1c17373ea008e980699a24b6",
    },
    "test": {
        "name": "test.zip",
        "bytes": 1_508_251_418,
        "checksum": "md5:d11e8379ba6f9ddf12e50a3d089ec1c2",
    },
}
RECEIPT_SCHEMA_VERSION = "qces_realdesed_acquisition_receipt_v1"


class AcquisitionError(RuntimeError):
    """Raised when publisher metadata, an archive, or extraction is unsafe."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(path: Path, algorithm: str, chunk_bytes: int = 4 << 20) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fetch_json(url: str) -> tuple[dict[str, Any], bytes]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "QCES-RealDESED-acquisition/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise AcquisitionError("Zenodo record response must be a JSON object")
    return payload, raw


def _publisher_file(record: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    expected = EXPECTED_FILES[split]
    files = record.get("files")
    if not isinstance(files, list):
        raise AcquisitionError("Zenodo record has no file list")
    matches = [item for item in files if item.get("key") == expected["name"]]
    if len(matches) != 1:
        raise AcquisitionError(
            f"expected exactly one Zenodo file named {expected['name']!r}"
        )
    item = matches[0]
    if item.get("size") != expected["bytes"]:
        raise AcquisitionError(f"publisher size drift for {expected['name']}")
    if item.get("checksum") != expected["checksum"]:
        raise AcquisitionError(f"publisher checksum drift for {expected['name']}")
    links = item.get("links")
    if not isinstance(links, Mapping) or not isinstance(links.get("self"), str):
        raise AcquisitionError(f"publisher download URL missing for {expected['name']}")
    return item


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=".partial",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "QCES-RealDESED-acquisition/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, temporary, length=4 << 20)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path.replace(destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def _validated_members(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    members = []
    seen = set()
    for member in archive.infolist():
        raw_name = member.filename
        if "\\" in raw_name:
            raise AcquisitionError(f"ZIP member uses a backslash: {raw_name!r}")
        path = PurePosixPath(raw_name)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(":" in part for part in path.parts)
        ):
            raise AcquisitionError(f"unsafe ZIP member path: {raw_name!r}")
        normalized = path.as_posix().rstrip("/")
        if normalized in seen:
            raise AcquisitionError(f"duplicate ZIP member path: {raw_name!r}")
        seen.add(normalized)
        unix_mode = member.external_attr >> 16
        if (unix_mode & 0o170000) == 0o120000:
            raise AcquisitionError(f"ZIP symlink is forbidden: {raw_name!r}")
        members.append(member)
    return tuple(members)


def _extract(archive_path: Path, output_dir: Path, split: str) -> Path:
    final_dir = output_dir / "raw" / split
    if final_dir.exists():
        return final_dir
    staging_parent = output_dir / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{split}.", dir=staging_parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _validated_members(archive)
            for member in members:
                archive.extract(member, staging_dir)
        candidates = [
            staging_dir / split,
            staging_dir,
        ]
        source_dir = next(
            (
                candidate
                for candidate in candidates
                if (candidate / "audio").is_dir()
                and (candidate / "metadata.csv").is_file()
                and (candidate / "annotations.csv").is_file()
            ),
            None,
        )
        if source_dir is None:
            raise AcquisitionError(
                "extracted split lacks audio/, metadata.csv, or annotations.csv"
            )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        source_dir.replace(final_dir)
        return final_dir
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _relative_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def acquire(
    *,
    split: str,
    output_dir: Path,
    publisher_record_path: Path | None = None,
    archive_path: Path | None = None,
    skip_extract: bool = False,
) -> dict[str, Any]:
    if split not in EXPECTED_FILES:
        raise ValueError(f"split must be one of {tuple(EXPECTED_FILES)}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance_dir = output_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    frozen_record_path = provenance_dir / f"zenodo_{ZENODO_RECORD_ID}.json"
    if publisher_record_path is None:
        if frozen_record_path.is_file():
            raw_record = frozen_record_path.read_bytes()
            record = json.loads(raw_record)
        else:
            record, raw_record = _fetch_json(ZENODO_API_URL)
    else:
        raw_record = publisher_record_path.read_bytes()
        record = json.loads(raw_record)
    if str(record.get("id")) != ZENODO_RECORD_ID:
        raise AcquisitionError("unexpected Zenodo record ID")
    publisher = _publisher_file(record, split)

    canonical_record = (_canonical_json(record) + "\n").encode("utf-8")
    if frozen_record_path.exists() and frozen_record_path.read_bytes() != canonical_record:
        raise AcquisitionError("frozen Zenodo record changed in place")
    if not frozen_record_path.exists():
        frozen_record_path.write_bytes(canonical_record)

    if archive_path is None:
        archive_path = output_dir / "downloads" / str(publisher["key"])
        _download(str(publisher["links"]["self"]), archive_path)
    else:
        archive_path = archive_path.resolve()
    expected_size = int(publisher["size"])
    if archive_path.stat().st_size != expected_size:
        raise AcquisitionError(
            f"archive byte size mismatch: {archive_path.stat().st_size} != {expected_size}"
        )
    expected_md5 = str(publisher["checksum"]).removeprefix("md5:")
    actual_md5 = _digest(archive_path, "md5")
    if actual_md5 != expected_md5:
        raise AcquisitionError(f"archive MD5 mismatch: {actual_md5} != {expected_md5}")
    archive_sha256 = _digest(archive_path, "sha256")

    extracted_dir = None
    extracted_files: list[Path] = []
    if not skip_extract:
        extracted_dir = _extract(archive_path, output_dir, split)
        extracted_files = list(_relative_files(extracted_dir))

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "RealDESED",
        "dataset_record_id": ZENODO_RECORD_ID,
        "dataset_record_url": f"https://zenodo.org/records/{ZENODO_RECORD_ID}",
        "dataset_record_api_sha256": _sha256_bytes(raw_record),
        "frozen_record_path": frozen_record_path.relative_to(output_dir).as_posix(),
        "split": split,
        "archive": {
            "path": archive_path.relative_to(output_dir).as_posix()
            if archive_path.is_relative_to(output_dir)
            else str(archive_path),
            "bytes": expected_size,
            "publisher_md5": expected_md5,
            "verified_md5": actual_md5,
            "sha256": archive_sha256,
        },
        "extraction": {
            "performed": not skip_extract,
            "path": (
                extracted_dir.relative_to(output_dir).as_posix()
                if extracted_dir is not None
                else None
            ),
            "file_count": len(extracted_files),
            "audio_file_count": sum(
                path.parent.name == "audio" for path in extracted_files
            ),
        },
        "license_contract": {
            "audio_and_corresponding_metadata": "per-file metadata.csv: CC0 or CC BY",
            "remaining_metadata_and_annotations": "CC BY 4.0",
            "source": "official RealDESED README and Zenodo record",
        },
    }
    receipt_dir = output_dir / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"acquisition_{split}.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        stable_fields = (
            "schema_version",
            "dataset",
            "dataset_record_id",
            "split",
            "archive",
            "extraction",
            "license_contract",
        )
        if any(existing.get(field) != receipt.get(field) for field in stable_fields):
            raise AcquisitionError("existing acquisition receipt conflicts with revalidation")
        return {**existing, "receipt_path": str(receipt_path)}
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**receipt, "receipt_path": str(receipt_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(EXPECTED_FILES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publisher-record", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--skip-extract", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = acquire(
            split=args.split,
            output_dir=args.output_dir,
            publisher_record_path=args.publisher_record,
            archive_path=args.archive,
            skip_extract=args.skip_extract,
        )
    except (AcquisitionError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"RealDESED acquisition failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
