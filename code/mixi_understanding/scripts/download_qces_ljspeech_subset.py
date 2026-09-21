#!/usr/bin/env python3
"""Download a small, receipt-backed LJSpeech subset through the HF viewer.

The original LJSpeech archive is about 2.7 GB.  QCES speech-event smoke data
only needs a few hundred utterances, so this downloader resolves individual
viewer assets and records the exact upstream revision and audio hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = "SeanSleat/lj_speech"
CONFIG = "main"
SPLIT = "train"


def _json_url(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "qces-ljspeech/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _bytes_url(url: str, attempts: int = 5) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "qces-ljspeech/1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError("empty audio payload")
            return payload
        except Exception as exc:  # network retry boundary
            error = exc
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"asset download failed after {attempts} attempts: {error}")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _rows(offset: int, length: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    payload = _json_url(f"https://datasets-server.huggingface.co/rows?{query}")
    if "error" in payload:
        raise RuntimeError(f"HF viewer error: {payload['error']}")
    return list(payload.get("rows") or [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "upstream/ljspeech_qces_single_speaker_v1",
    )
    parser.add_argument("--num-items", type=int, default=280)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.num_items < 1 or args.workers < 1 or args.offset < 0:
        raise SystemExit("num-items/workers must be positive and offset non-negative")

    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    rows: list[dict[str, Any]] = []
    for page_offset in range(args.offset, args.offset + args.num_items, 100):
        length = min(100, args.offset + args.num_items - page_offset)
        rows.extend(_rows(page_offset, length))
    if len(rows) != args.num_items:
        raise RuntimeError(f"viewer returned {len(rows)} != {args.num_items} rows")

    dataset_info = _json_url(
        f"https://huggingface.co/api/datasets/{urllib.parse.quote(DATASET, safe='/')}"
    )
    revision = str(dataset_info.get("sha") or "")

    def download(raw: dict[str, Any]) -> dict[str, Any]:
        row_index = int(raw["row_idx"])
        row = dict(raw["row"])
        utterance_id = str(row["id"])
        audio_cell = row.get("audio") or []
        if not isinstance(audio_cell, list) or not audio_cell:
            raise RuntimeError(f"missing viewer audio asset for {utterance_id}")
        url = str(audio_cell[0]["src"])
        path = audio_dir / f"{utterance_id}.wav"
        payload = path.read_bytes() if path.is_file() else _bytes_url(url)
        info = sf.info(io.BytesIO(payload))
        if info.frames <= 0 or info.samplerate <= 0 or info.channels != 1:
            raise RuntimeError(f"invalid mono speech audio for {utterance_id}: {info}")
        if not path.is_file():
            _atomic_bytes(path, payload)
        return {
            "format": "qces_ljspeech_utterance_v1",
            "utterance_id": utterance_id,
            "speaker_id": "LJ",
            "speaker_group": "female",
            "text": str(row.get("text") or "").strip(),
            "normalized_text": str(row.get("normalized_text") or row.get("text") or "").strip(),
            "audio_path": str(path.relative_to(PROJECT_ROOT)),
            "audio_sha256": hashlib.sha256(payload).hexdigest(),
            "audio_bytes": len(payload),
            "sample_rate": int(info.samplerate),
            "duration_seconds": float(info.duration),
            "upstream_row_index": row_index,
            "upstream_dataset": DATASET,
            "upstream_revision": revision,
            "upstream_split": SPLIT,
            "license": "Unlicense",
        }

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, row) for row in rows]
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"downloaded {index}/{len(futures)}", flush=True)
    completed.sort(key=lambda row: int(row["upstream_row_index"]))

    manifest_path = output_dir / "ljspeech_manifest.jsonl"
    _atomic_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in completed),
    )
    receipt = {
        "format": "qces_ljspeech_subset_receipt_v1",
        "complete": True,
        "dataset": DATASET,
        "revision": revision,
        "config": CONFIG,
        "split": SPLIT,
        "offset": args.offset,
        "items": len(completed),
        "speaker_count": 1,
        "speaker_group": "female",
        "total_duration_seconds": sum(float(row["duration_seconds"]) for row in completed),
        "total_audio_bytes": sum(int(row["audio_bytes"]) for row in completed),
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output_dir / "download_receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

