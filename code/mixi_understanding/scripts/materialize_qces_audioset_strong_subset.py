#!/usr/bin/env python3
"""Materialize a targeted AudioSet-Strong subset by streaming HF rows.

This does not download the full AudioSet-Strong parquet collection.  It streams
rows from ``enyoukai/AudioSet-Strong``, keeps only clips whose strong events
match the selected QCES ontology, writes the audio bytes, and emits detector
manifests compatible with the BEATs-Strong training script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import soundfile as sf
from datasets import Audio, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ONTOLOGY_TSV = PROJECT_ROOT / "outputs/qces_multisource_200_ontology_v1/ontology_200_multisource.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_audioset_strong_subset_v1"
FORMAT = "qces_audioset_strong_subset_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology-tsv", type=Path, default=DEFAULT_ONTOLOGY_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-dataset", default="enyoukai/AudioSet-Strong")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--train-quota-per-label", type=int, default=40)
    parser.add_argument("--test-quota-per-label", type=int, default=10)
    parser.add_argument("--max-clips-per-split", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Write partial manifest/summary every N kept clips; 0 disables.",
    )
    parser.add_argument("--overwrite-manifests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.train_quota_per_label <= 0 or args.test_quota_per_label <= 0:
        parser.error("quotas must be positive")
    if args.max_clips_per_split < 0:
        parser.error("--max-clips-per-split must be >= 0")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be >= 0")
    return args


def safe_label(label: str) -> str:
    text = str(label).strip()
    text = text.replace("&", "and")
    text = re.sub(r",\s*", "_and_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.replace("/", "_")
    text = text.replace('"', "")
    return text


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite-manifests")
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


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite-manifests")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_selected_audioset_labels(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label = str(row["label"])
            if int(row.get("audioset_train_clips") or 0) <= 0:
                continue
            display = str(row["display_name"])
            labels[label] = {
                "label": label,
                "display_name": display,
                "audioset_mid": str(row.get("audioset_mid") or ""),
            }
    if not labels:
        raise ValueError(f"no AudioSet-positive labels in ontology: {path}")
    return labels


def extension_from_path(path: str | None) -> str:
    if not path:
        return ".flac"
    suffix = Path(path).suffix.lower()
    return suffix if suffix else ".flac"


def row_events(row: Mapping[str, Any], selected: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_event in row.get("events") or []:
        if not isinstance(raw_event, dict):
            continue
        display = str(raw_event.get("event_name") or "").strip()
        label = safe_label(display)
        if label not in selected:
            continue
        onset = float(raw_event.get("start", 0.0))
        offset = float(raw_event.get("end", 0.0))
        if offset <= onset:
            continue
        events.append(
            {
                "event_kind": "semantic",
                "label": label,
                "display_name": selected[label]["display_name"],
                "audioset_mid": selected[label]["audioset_mid"],
                "onset_seconds": max(0.0, onset),
                "offset_seconds": min(10.0, offset),
            }
        )
    events.sort(
        key=lambda item: (
            float(item["onset_seconds"]),
            float(item["offset_seconds"]),
            str(item["label"]),
        )
    )
    return events


def materialize_split(
    *,
    hf_dataset: str,
    hf_split: str,
    output_dir: Path,
    selected: Mapping[str, Mapping[str, str]],
    quota_per_label: int,
    max_clips: int,
    checkpoint_every: int,
    dry_run: bool,
    overwrite_manifests: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset(hf_dataset, split=hf_split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    counts: Counter[str] = Counter()
    manifests: list[dict[str, Any]] = []
    seen = 0
    kept = 0
    for row in dataset:
        seen += 1
        events = row_events(row, selected)
        if not events:
            continue
        helpful_labels = sorted(
            {
                str(event["label"])
                for event in events
                if counts[str(event["label"])] < quota_per_label
            }
        )
        if not helpful_labels:
            continue
        audio = row["audio"]
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        audio_path = audio.get("path") if isinstance(audio, dict) else None
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            continue
        video_id = str(row.get("video_id") or f"{hf_split}_{seen:08d}")
        suffix = extension_from_path(audio_path)
        out_audio = output_dir / "audio" / hf_split / f"{video_id}{suffix}"
        if not dry_run and not out_audio.exists():
            atomic_bytes(out_audio, audio_bytes)
        sample_rate = 0
        duration = 10.0
        if not dry_run:
            try:
                info = sf.info(out_audio)
                sample_rate = int(info.samplerate)
                duration = float(info.duration)
            except Exception:
                sample_rate = 0
                duration = 10.0
        for label in helpful_labels:
            counts[label] += 1
        labels = sorted({str(event["label"]) for event in events})
        for index, event in enumerate(events):
            event["event_id"] = f"{video_id}_{index:03d}_{event['label']}"
        manifests.append(
            {
                "format": FORMAT,
                "scene_id": f"audioset_strong_{hf_split}_{video_id}",
                "source_route": "audioset_strong_hf_stream",
                "hf_dataset": hf_dataset,
                "hf_split": hf_split,
                "video_id": video_id,
                "mixture_path": project_relative(out_audio),
                "audio_sha256": sha256_bytes(audio_bytes),
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "labels": labels,
                "events": events,
            }
        )
        kept += 1
        if kept % 250 == 0:
            print(
                f"split={hf_split} kept={kept} seen={seen} "
                f"covered_labels={sum(1 for label in selected if counts[label] > 0)}/{len(selected)}",
                flush=True,
            )
        if checkpoint_every and kept % checkpoint_every == 0:
            partial_manifest = output_dir / f"audioset_strong_detector_manifest_{hf_split}.partial.jsonl"
            partial_summary = output_dir / f"audioset_strong_subset_{hf_split}.partial_summary.json"
            atomic_jsonl(
                partial_manifest,
                manifests,
                overwrite=overwrite_manifests or partial_manifest.exists(),
            )
            atomic_json(
                partial_summary,
                {
                    "format": FORMAT,
                    "partial": True,
                    "hf_split": hf_split,
                    "seen_rows": seen,
                    "kept_clips": kept,
                    "quota_per_label": quota_per_label,
                    "labels_with_at_least_one_clip": sum(1 for label in selected if counts[label] > 0),
                    "labels_reaching_quota": sum(1 for label in selected if counts[label] >= quota_per_label),
                    "label_counts": dict(sorted(counts.items())),
                    "manifest_path": str(partial_manifest.resolve()),
                },
                overwrite=overwrite_manifests or partial_summary.exists(),
            )
        if max_clips and kept >= max_clips:
            break
        if all(counts[label] >= quota_per_label for label in selected):
            break
    summary = {
        "hf_split": hf_split,
        "seen_rows": seen,
        "kept_clips": kept,
        "quota_per_label": quota_per_label,
        "labels_with_at_least_one_clip": sum(1 for label in selected if counts[label] > 0),
        "labels_reaching_quota": sum(1 for label in selected if counts[label] >= quota_per_label),
        "label_counts": dict(sorted(counts.items())),
        "dry_run": dry_run,
    }
    return manifests, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_selected_audioset_labels(args.ontology_tsv.resolve())
    split_summaries: dict[str, Any] = {}
    manifest_paths: dict[str, str] = {}
    for split in args.splits:
        if split not in {"train", "test"}:
            raise ValueError("HF split must be train or test")
        quota = args.train_quota_per_label if split == "train" else args.test_quota_per_label
        manifests, summary = materialize_split(
            hf_dataset=args.hf_dataset,
            hf_split=split,
            output_dir=output_dir,
            selected=selected,
            quota_per_label=quota,
            max_clips=args.max_clips_per_split,
            checkpoint_every=args.checkpoint_every,
            dry_run=args.dry_run,
            overwrite_manifests=args.overwrite_manifests,
        )
        manifest_name = f"audioset_strong_detector_manifest_{split}.jsonl"
        manifest_path = output_dir / manifest_name
        atomic_jsonl(
            manifest_path,
            manifests,
            overwrite=args.overwrite_manifests,
        )
        split_summaries[split] = summary
        manifest_paths[split] = str(manifest_path.resolve())
    final_summary = {
        "format": FORMAT,
        "hf_dataset": args.hf_dataset,
        "ontology_tsv": str(args.ontology_tsv.resolve()),
        "output_dir": str(output_dir),
        "selected_audioset_labels": len(selected),
        "manifest_paths": manifest_paths,
        "splits": split_summaries,
    }
    atomic_json(
        output_dir / "audioset_strong_subset_summary.json",
        final_summary,
        overwrite=args.overwrite_manifests,
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
    # Datasets/torchcodec currently sometimes crashes during interpreter
    # finalization after streaming Audio(decode=False). Explicit exit keeps the
    # completed manifests/files usable and avoids a false failure status.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
