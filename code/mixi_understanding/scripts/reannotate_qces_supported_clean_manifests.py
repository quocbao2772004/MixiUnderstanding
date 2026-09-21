#!/usr/bin/env python3
"""Re-annotate clean manifests with the supported 200-class ontology.

Historical AudioSet manifests were filtered by an older ontology and therefore
omit valid labels even when the audio is already present locally.  This
sidecar restores all official strong events for selected labels while keeping
non-selected events as explicit context annotations.  Split assignments and
audio identities are never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from mixi_understanding.qces.benchmark_integrity import audit_split_overlaps
from mixi_understanding.qces.supported_ontology import load_strong_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_clean_detector_protocol_v1_current",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1/ontology_200_supported.txt",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=PROJECT_ROOT / "code/baseline/PretrainedSED/hf_dataset_gen/metadata",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_supported_clean_protocol_v2_current_audio",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_text(path: Path, value: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strong_event(event: Any, index: int, *, semantic: bool) -> dict[str, Any]:
    return {
        "event_id": f"{event.video_id}_{index:04d}_{event.label}",
        "event_kind": "semantic" if semantic else "context_unscored",
        "label": event.label,
        "display_name": event.display_name,
        "audioset_mid": event.mid,
        "segment_id": event.segment_id,
        "onset_seconds": float(event.onset_seconds),
        "offset_seconds": float(event.offset_seconds),
    }


def _reannotate(
    rows: Iterable[Mapping[str, Any]],
    *,
    selected: set[str],
    strong_events: Mapping[str, Mapping[str, list[Any]]],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    output: list[dict[str, Any]] = []
    support: Counter[str] = Counter()
    restored: Counter[str] = Counter()
    for raw in rows:
        row = dict(raw)
        source = str(row.get("protocol_source") or row.get("source_route") or "")
        old_pairs = {
            (str(event.get("label") or ""), float(event.get("onset_seconds", 0.0)), float(event.get("offset_seconds", 0.0)))
            for event in row.get("events") or ()
        }
        if "audioset" in source.lower():
            video_id = str(row.get("video_id") or "")
            metadata_split = "eval" if str(row.get("hf_split") or "").lower() == "test" else "train"
            official = list(strong_events[metadata_split].get(video_id, ()))
            selected_events = [event for event in official if event.label in selected]
            context_events = [event for event in official if event.label not in selected]
            row["events"] = [
                _strong_event(event, index, semantic=True)
                for index, event in enumerate(selected_events)
            ]
            row["context_events"] = [
                _strong_event(event, index, semantic=False)
                for index, event in enumerate(context_events)
            ]
            row["labels"] = sorted({event.label for event in selected_events})
            row["annotation_source"] = "official_audioset_strong_full_metadata"
            for event in selected_events:
                support[event.label] += 1
                key = (event.label, float(event.onset_seconds), float(event.offset_seconds))
                restored[event.label] += int(key not in old_pairs)
        else:
            original = list(row.get("events") or ())
            row["events"] = [
                {**event, "event_kind": "semantic"}
                for event in original
                if str(event.get("label") or "") in selected
            ]
            row["context_events"] = [
                {**event, "event_kind": "context_unscored"}
                for event in original
                if str(event.get("label") or "") not in selected
            ]
            row["labels"] = sorted({str(event["label"]) for event in row["events"]})
            for event in row["events"]:
                support[str(event["label"])] += 1
        row["ontology_version"] = "qces_supported_ontology_200_v1"
        output.append(row)
    return output, support, restored


def main() -> int:
    args = parse_args()
    labels = [
        line.strip()
        for line in args.ontology.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != 200 or len(set(labels)) != 200:
        raise SystemExit("ontology must contain exactly 200 unique labels")
    selected = set(labels)
    _, strong_events = load_strong_metadata(args.metadata_dir)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_support: dict[str, Counter[str]] = {}
    split_restored: dict[str, Counter[str]] = {}
    for split in ("train", "dev", "test"):
        rows, support, restored = _reannotate(
            _read_jsonl(args.input_root / f"detector_manifest_{split}.jsonl"),
            selected=selected,
            strong_events=strong_events,
        )
        split_rows[split] = rows
        split_support[split] = support
        split_restored[split] = restored

    overlap = audit_split_overlaps(split_rows)
    hard_overlap = int(overlap.get("hard_overlap_count", 0))
    if hard_overlap:
        raise RuntimeError(f"re-annotation changed/leaked identities: hard overlap={hard_overlap}")
    output_paths = {
        split: args.output_root / f"detector_manifest_{split}.jsonl"
        for split in ("train", "dev", "test")
    }
    for split, path in output_paths.items():
        text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in split_rows[split]
        )
        _atomic_text(path, text, overwrite=args.overwrite)
    _atomic_text(
        args.output_root / "ontology_200_supported.txt",
        "\n".join(labels) + "\n",
        overwrite=args.overwrite,
    )
    receipt = {
        "format": "qces_supported_clean_protocol_v2_current_audio",
        "ontology_labels": len(labels),
        "cross_split_hard_overlap": hard_overlap,
        "split_rows": {split: len(rows) for split, rows in split_rows.items()},
        "classes_with_positive": {
            split: sum(split_support[split][label] > 0 for label in labels)
            for split in split_rows
        },
        "restored_events": {
            split: sum(split_restored[split].values()) for split in split_rows
        },
        "output_sha256": {
            split: _sha256(path) for split, path in output_paths.items()
        },
    }
    _atomic_text(
        args.output_root / "reannotation_receipt.json",
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
