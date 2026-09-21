#!/usr/bin/env python3
"""Build a 200-class detector ontology from FUSS/FSD50K + AudioSet-Strong.

The previous detector artifact was intentionally conservative and used only
locally verified FUSS/FSD50K positives.  That route is too narrow for 200
classes.  This script adds AudioSet-Strong at the planning layer:

* read local AudioSet-Strong timestamp CSVs;
* map AudioSet display names to QCES-safe labels;
* merge coverage with the existing FUSS/FSD50K sourcebank coverage;
* keep covered labels from the existing QCES label bank;
* fill the remaining slots from AudioSet-Strong labels with strong timestamps;
* write clip-level sampling plans for downloading/streaming only the needed
  AudioSet-Strong clips instead of materializing the full dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AS_META = PROJECT_ROOT / "code/baseline/PretrainedSED/hf_dataset_gen/metadata"
DEFAULT_FUSS_COVERAGE = PROJECT_ROOT / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1/label_coverage.json"
DEFAULT_REQUESTED_BANK = PROJECT_ROOT / "outputs/qces_open_event_v2/label_bank_200.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_multisource_200_ontology_v1"

FORMAT = "qces_multisource_200_ontology_v1"
GENERIC_LABELS = {
    "Background_noise",
    "Channel_and_environment_and_background",
    "Environmental_noise",
    "Generic_impact_sounds",
    "Human_sounds",
    "Mechanisms",
    "Music",
    "Noise",
    "Silence",
    "Sound_effect",
    "Source-ambiguous_sounds",
    "Unmodified_field_recording",
}


@dataclass
class CombinedCoverage:
    label: str
    display_name: str
    requested_label: str | None
    in_requested_bank: bool
    audioset_mid: str | None
    audioset_train_clips: int
    audioset_eval_clips: int
    audioset_train_events: int
    audioset_eval_events: int
    audioset_train_seconds: float
    audioset_eval_seconds: float
    fuss_train_sources: int
    fuss_val_sources: int
    fuss_test_sources: int
    fuss_train_seconds: float
    train_positive: bool
    ready_weak: bool
    ready_medium: bool
    ready_strong: bool
    source_route: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audioset-metadata-dir", type=Path, default=DEFAULT_AS_META)
    parser.add_argument("--fuss-coverage-json", type=Path, default=DEFAULT_FUSS_COVERAGE)
    parser.add_argument("--requested-label-bank", type=Path, default=DEFAULT_REQUESTED_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-labels", type=int, default=200)
    parser.add_argument("--train-quota-per-label", type=int, default=80)
    parser.add_argument("--eval-quota-per-label", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.target_labels <= 0:
        parser.error("--target-labels must be positive")
    if args.train_quota_per_label <= 0 or args.eval_quota_per_label <= 0:
        parser.error("quotas must be positive")
    return args


def safe_label(label: str) -> str:
    text = str(label).strip()
    text = text.replace("&", "and")
    text = re.sub(r",\s*", "_and_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.replace("/", "_")
    text = text.replace('"', "")
    return text


def stable_key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
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


def write_lines(path: Path, lines: Sequence[str], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
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


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["label"]
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_requested_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        label = raw.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def load_audioset_mid_map(metadata_dir: Path) -> dict[str, str]:
    path = metadata_dir / "class_labels_indices_strong.csv"
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError(f"bad AudioSet class row: {row!r}")
            result[row[0]] = row[1]
    return result


def load_audioset_coverage(
    metadata_dir: Path,
    mid_to_display: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    coverage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "display_name": "",
            "mid": None,
            "train_clips": set(),
            "eval_clips": set(),
            "train_events": 0,
            "eval_events": 0,
            "train_seconds": 0.0,
            "eval_seconds": 0.0,
        }
    )
    clip_events: dict[str, dict[str, list[dict[str, Any]]]] = {
        "train": defaultdict(list),
        "eval": defaultdict(list),
    }
    for split, filename in (
        ("train", "audioset_train_strong.csv"),
        ("eval", "audioset_eval_strong.csv"),
    ):
        with (metadata_dir / filename).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"segment_id", "start_time_seconds", "end_time_seconds", "label"}
            if set(reader.fieldnames or ()) != required:
                raise ValueError(f"bad header for {filename}: {reader.fieldnames}")
            for row in reader:
                mid = row["label"]
                display = mid_to_display.get(mid)
                if display is None:
                    raise ValueError(f"missing MID in class map: {mid}")
                label = safe_label(display)
                onset = float(row["start_time_seconds"])
                offset = float(row["end_time_seconds"])
                if offset <= onset:
                    continue
                item = coverage[label]
                item["display_name"] = display
                item["mid"] = mid
                item[f"{split}_clips"].add(row["segment_id"])
                item[f"{split}_events"] += 1
                item[f"{split}_seconds"] += max(0.0, offset - onset)
                clip_events[split][row["segment_id"]].append(
                    {
                        "label": label,
                        "display_name": display,
                        "mid": mid,
                        "onset_seconds": onset,
                        "offset_seconds": offset,
                    }
                )
    compact: dict[str, dict[str, Any]] = {}
    for label, row in coverage.items():
        compact[label] = {
            **row,
            "train_clips": len(row["train_clips"]),
            "eval_clips": len(row["eval_clips"]),
        }
    return compact, clip_events


def load_fuss_coverage(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows", []):
        label = str(row["label"])
        result[label] = row
    return result


def combine_coverage(
    *,
    requested: Sequence[str],
    audioset: Mapping[str, Mapping[str, Any]],
    fuss: Mapping[str, Mapping[str, Any]],
) -> list[CombinedCoverage]:
    requested_by_safe = {safe_label(label): label for label in requested}
    labels = sorted(set(audioset) | set(fuss) | set(requested_by_safe))
    combined: list[CombinedCoverage] = []
    for label in labels:
        as_row = audioset.get(label, {})
        fuss_row = fuss.get(label, {})
        as_train = int(as_row.get("train_clips", 0))
        as_eval = int(as_row.get("eval_clips", 0))
        fuss_train = int(fuss_row.get("train_sources_ge_event", 0))
        fuss_val = int(fuss_row.get("validation_sources_ge_event", 0))
        fuss_test = int(fuss_row.get("eval_sources_ge_event", 0))
        train_total = as_train + fuss_train
        eval_total = as_eval + fuss_val + fuss_test
        train_positive = train_total >= 1
        ready_weak = train_total >= 4 and eval_total >= 2
        ready_medium = train_total >= 10 and eval_total >= 4
        ready_strong = train_total >= 20 and eval_total >= 4
        if as_row and fuss_row:
            route = "audioset_strong+fuss_fsd50k"
        elif as_row:
            route = "audioset_strong"
        elif fuss_row:
            route = "fuss_fsd50k"
        else:
            route = "missing"
        requested_label = requested_by_safe.get(label)
        display_name = str(as_row.get("display_name") or fuss_row.get("label") or requested_label or label)
        combined.append(
            CombinedCoverage(
                label=label,
                display_name=display_name,
                requested_label=requested_label,
                in_requested_bank=requested_label is not None,
                audioset_mid=as_row.get("mid"),
                audioset_train_clips=as_train,
                audioset_eval_clips=as_eval,
                audioset_train_events=int(as_row.get("train_events", 0)),
                audioset_eval_events=int(as_row.get("eval_events", 0)),
                audioset_train_seconds=float(as_row.get("train_seconds", 0.0)),
                audioset_eval_seconds=float(as_row.get("eval_seconds", 0.0)),
                fuss_train_sources=fuss_train,
                fuss_val_sources=fuss_val,
                fuss_test_sources=fuss_test,
                fuss_train_seconds=float(fuss_row.get("train_active_seconds_ge_event", 0.0)),
                train_positive=train_positive,
                ready_weak=ready_weak,
                ready_medium=ready_medium,
                ready_strong=ready_strong,
                source_route=route,
            )
        )
    return sorted(combined, key=lambda row: row.label)


def select_ontology(
    combined: Sequence[CombinedCoverage],
    requested: Sequence[str],
    *,
    target_labels: int,
) -> list[CombinedCoverage]:
    by_label = {row.label: row for row in combined}
    selected: list[CombinedCoverage] = []
    selected_labels: set[str] = set()
    for requested_label in requested:
        label = safe_label(requested_label)
        row = by_label.get(label)
        if row is not None and row.train_positive and label not in selected_labels:
            selected.append(row)
            selected_labels.add(label)
        if len(selected) == target_labels:
            return selected

    fillers = [
        row
        for row in combined
        if row.label not in selected_labels and row.train_positive
    ]
    fillers.sort(
        key=lambda row: (
            1 if row.label in GENERIC_LABELS else 0,
            0 if row.ready_strong else 1,
            0 if row.ready_medium else 1,
            0 if row.ready_weak else 1,
            -row.audioset_train_clips - row.fuss_train_sources,
            -row.audioset_eval_clips - row.fuss_val_sources - row.fuss_test_sources,
            row.label,
        )
    )
    for row in fillers:
        selected.append(row)
        selected_labels.add(row.label)
        if len(selected) == target_labels:
            return selected
    raise RuntimeError(
        f"only {len(selected)} train-positive labels available; target={target_labels}"
    )


def sample_audioset_clips(
    clip_events: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_labels: set[str],
    *,
    split: str,
    quota_per_label: int,
) -> tuple[list[dict[str, Any]], Counter]:
    counts: Counter[str] = Counter()
    selected_rows: list[dict[str, Any]] = []
    for segment_id in sorted(clip_events, key=lambda value: stable_key(split, value)):
        events = [
            dict(event)
            for event in clip_events[segment_id]
            if str(event["label"]) in selected_labels
        ]
        if not events:
            continue
        helpful = any(counts[str(event["label"])] < quota_per_label for event in events)
        if not helpful:
            continue
        labels = sorted({str(event["label"]) for event in events})
        for label in labels:
            if counts[label] < quota_per_label:
                counts[label] += 1
        selected_rows.append(
            {
                "format": FORMAT,
                "source_route": "audioset_strong_hf_stream_plan",
                "hf_dataset": "enyoukai/AudioSet-Strong",
                "split": split,
                "segment_id": segment_id,
                "video_id": segment_id.rsplit("_", 1)[0],
                "duration_seconds": 10.0,
                "labels": labels,
                "events": sorted(
                    events,
                    key=lambda item: (
                        float(item["onset_seconds"]),
                        float(item["offset_seconds"]),
                        str(item["label"]),
                    ),
                ),
            }
        )
        if all(counts[label] >= quota_per_label for label in selected_labels if label in counts):
            # Do not stop here: labels not encountered yet may still be present
            # later, but this check keeps the condition explicit.
            pass
    return selected_rows, counts


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = load_requested_labels(args.requested_label_bank.resolve())
    mid_to_display = load_audioset_mid_map(args.audioset_metadata_dir.resolve())
    audioset, clip_events = load_audioset_coverage(args.audioset_metadata_dir.resolve(), mid_to_display)
    fuss = load_fuss_coverage(args.fuss_coverage_json.resolve())
    combined = combine_coverage(requested=requested, audioset=audioset, fuss=fuss)
    selected = select_ontology(combined, requested, target_labels=args.target_labels)
    selected_labels = {row.label for row in selected}

    train_plan, train_counts = sample_audioset_clips(
        clip_events["train"],
        selected_labels,
        split="train",
        quota_per_label=args.train_quota_per_label,
    )
    eval_plan, eval_counts = sample_audioset_clips(
        clip_events["eval"],
        selected_labels,
        split="eval",
        quota_per_label=args.eval_quota_per_label,
    )

    coverage_rows = [asdict(row) for row in combined]
    selected_rows = [asdict(row) for row in selected]
    write_tsv(output_dir / "multisource_label_coverage.tsv", coverage_rows, overwrite=args.overwrite)
    write_tsv(output_dir / "ontology_200_multisource.tsv", selected_rows, overwrite=args.overwrite)
    write_lines(output_dir / "ontology_200_multisource.txt", [row.label for row in selected], overwrite=args.overwrite)
    write_jsonl(output_dir / "audioset_strong_sample_plan_train.jsonl", train_plan, overwrite=args.overwrite)
    write_jsonl(output_dir / "audioset_strong_sample_plan_eval.jsonl", eval_plan, overwrite=args.overwrite)

    requested_covered = [
        row for row in combined if row.in_requested_bank and row.train_positive
    ]
    requested_missing = [
        row.requested_label
        for row in combined
        if row.in_requested_bank and not row.train_positive and row.requested_label is not None
    ]
    selected_route_counts = Counter(row.source_route for row in selected)
    selected_ready_counts = {
        "ready_weak": sum(row.ready_weak for row in selected),
        "ready_medium": sum(row.ready_medium for row in selected),
        "ready_strong": sum(row.ready_strong for row in selected),
    }
    audioset_selected_labels = {row.label for row in selected if row.audioset_train_clips > 0}
    summary = {
        "format": FORMAT,
        "target_labels": args.target_labels,
        "requested_label_bank": str(args.requested_label_bank.resolve()),
        "counts": {
            "requested_labels": len(requested),
            "audioset_strong_labels": len(audioset),
            "audioset_strong_train_positive_labels": sum(
                int(row["train_clips"] > 0) for row in audioset.values()
            ),
            "fuss_train_positive_labels": sum(
                int(bool(row.get("train_positive"))) for row in fuss.values()
            ),
            "combined_train_positive_labels": sum(row.train_positive for row in combined),
            "requested_train_positive_after_union": len(requested_covered),
            "requested_missing_after_union": len(requested_missing),
            "selected_labels": len(selected),
            "selected_audioset_train_positive_labels": len(audioset_selected_labels),
            "selected_route_counts": dict(selected_route_counts),
            **selected_ready_counts,
            "audioset_train_plan_clips": len(train_plan),
            "audioset_eval_plan_clips": len(eval_plan),
        },
        "quotas": {
            "audioset_train_quota_per_label": args.train_quota_per_label,
            "audioset_eval_quota_per_label": args.eval_quota_per_label,
        },
        "requested_missing_after_union": requested_missing,
        "new_fill_labels": [
            asdict(row)
            for row in selected
            if not row.in_requested_bank
        ],
        "paths": {
            "ontology": str((output_dir / "ontology_200_multisource.txt").resolve()),
            "ontology_tsv": str((output_dir / "ontology_200_multisource.tsv").resolve()),
            "coverage_tsv": str((output_dir / "multisource_label_coverage.tsv").resolve()),
            "audioset_train_plan": str((output_dir / "audioset_strong_sample_plan_train.jsonl").resolve()),
            "audioset_eval_plan": str((output_dir / "audioset_strong_sample_plan_eval.jsonl").resolve()),
        },
        "notes": [
            "AudioSet-Strong provides timestamps but not isolated stems.",
            "The sample plans list clips to stream/download; they do not materialize the full HF dataset.",
            "FUSS/FSD50K remains useful for isolated-source synthetic mixtures and clean target stems.",
        ],
    }
    write_json(output_dir / "multisource_200_summary.json", summary, overwrite=args.overwrite)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2), flush=True)
    print(f"multisource ontology ready: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
