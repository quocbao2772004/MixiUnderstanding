#!/usr/bin/env python3
"""Render oracle pointer diagnostics for QCES external AudioQA audits.

The normal QCES AudioQA evaluator supports only a fixed condition name
(`predicted_evidence`).  This script creates evaluator-compatible prediction
roots and companion manifests where `predicted_evidence.wav` is one diagnostic
intervention:

* target_only: only oracle answer-event stems are audible.
* context_original: mixture context around oracle anchor + answer spans is audible.
* context_span_text: same audio as context_original, but the question states the
  answer span times.
* context_marked: same audio as context_original, plus a short beep immediately
  before the oracle answer onset; the question refers to the marker.

These are diagnostics, not deployable systems.  They test whether a frozen LALM
fails because it cannot localize the answer inside evidence, or because it
cannot recognize the sound even when localized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


MODES = (
    "target_only",
    "answer_crop_front_norm",
    "context_original",
    "context_span_text",
    "context_marked",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Diagnostic modes to render. Defaults to all modes.",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--padding-seconds", type=float, default=0.20)
    parser.add_argument("--fade-milliseconds", type=float, default=10.0)
    parser.add_argument("--marker-frequency-hz", type=float, default=1000.0)
    parser.add_argument("--marker-duration-seconds", type=float, default=0.10)
    parser.add_argument("--marker-amplitude", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"unsafe path outside dataset root: {path}")
    return path


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        waveform = waveform.mean(axis=1, keepdims=True)
    mono = np.ascontiguousarray(waveform[:, 0])
    if not np.isfinite(mono).all():
        raise ValueError(f"non-finite audio values: {path}")
    return mono, int(sample_rate)


def event_map(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(event["event_id"]): event for event in record.get("events", [])}


def intervals_from_event_ids(
    record: Mapping[str, Any], event_ids: Sequence[str]
) -> list[tuple[float, float]]:
    events = event_map(record)
    spans: list[tuple[float, float]] = []
    for event_id in event_ids:
        event = events.get(str(event_id))
        if event is None:
            continue
        spans.append((float(event["onset_seconds"]), float(event["offset_seconds"])))
    return spans


def coerce_intervals(value: Any) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                spans.append((float(item[0]), float(item[1])))
    return spans


def evidence_intervals(record: Mapping[str, Any]) -> list[tuple[float, float]]:
    spans = coerce_intervals(record.get("anchor_intervals"))
    spans.extend(coerce_intervals(record.get("answer_intervals")))
    if not spans:
        spans = intervals_from_event_ids(
            record, [str(x) for x in record.get("evidence_event_ids", [])]
        )
    return spans


def answer_intervals(record: Mapping[str, Any]) -> list[tuple[float, float]]:
    spans = coerce_intervals(record.get("answer_intervals"))
    if not spans:
        spans = intervals_from_event_ids(
            record, [str(x) for x in record.get("answer_event_ids", [])]
        )
    return spans


def apply_fade(mask: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    fade = int(round(sample_rate * fade_ms / 1000.0))
    if fade <= 1:
        return mask
    edges = np.flatnonzero(np.diff(np.r_[0.0, mask, 0.0]) != 0)
    if len(edges) % 2:
        return mask
    out = mask.copy()
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    for start, end in zip(edges[0::2], edges[1::2]):
        rise_end = min(end, start + fade)
        rise_len = rise_end - start
        if rise_len > 0:
            out[start:rise_end] = np.minimum(out[start:rise_end], ramp[:rise_len])
        fall_start = max(start, end - fade)
        fall_len = end - fall_start
        if fall_len > 0:
            out[fall_start:end] = np.minimum(
                out[fall_start:end], ramp[:fall_len][::-1]
            )
    return out


def context_mask(
    spans: Sequence[tuple[float, float]],
    *,
    num_samples: int,
    sample_rate: int,
    padding_seconds: float,
    fade_milliseconds: float,
) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=np.float32)
    if not spans:
        return mask
    start = max(0.0, min(a for a, _ in spans) - padding_seconds)
    end = min(num_samples / sample_rate, max(b for _, b in spans) + padding_seconds)
    a = max(0, min(num_samples, int(round(start * sample_rate))))
    b = max(0, min(num_samples, int(round(end * sample_rate))))
    if b > a:
        mask[a:b] = 1.0
    return apply_fade(mask, sample_rate, fade_milliseconds)


def sum_stems(
    record: Mapping[str, Any],
    dataset_root: Path,
    event_ids: Sequence[str],
    *,
    num_samples: int,
    sample_rate: int,
) -> np.ndarray:
    if not event_ids:
        return np.zeros(num_samples, dtype=np.float32)
    events = event_map(record)
    out = np.zeros(num_samples, dtype=np.float32)
    for event_id in event_ids:
        event = events[str(event_id)]
        path = safe_child(dataset_root, str(event["stem_path"]))
        stem, stem_rate = read_mono(path)
        if stem_rate != sample_rate:
            raise ValueError(f"sample-rate mismatch at {path}: {stem_rate}")
        if stem.shape[0] != num_samples:
            raise ValueError(f"sample-count mismatch at {path}: {stem.shape[0]}")
        out += stem
    return out


def add_marker(
    evidence: np.ndarray,
    *,
    answer_spans: Sequence[tuple[float, float]],
    sample_rate: int,
    frequency_hz: float,
    duration_seconds: float,
    amplitude: float,
) -> np.ndarray:
    if not answer_spans:
        return evidence
    out = evidence.copy()
    answer_start = min(start for start, _ in answer_spans)
    marker_samples = max(1, int(round(duration_seconds * sample_rate)))
    marker_end = max(0, min(out.shape[0], int(round(answer_start * sample_rate))))
    marker_start = max(0, marker_end - marker_samples)
    if marker_end <= marker_start:
        marker_start = max(0, min(out.shape[0], int(round(answer_start * sample_rate))))
        marker_end = min(out.shape[0], marker_start + marker_samples)
    if marker_end <= marker_start:
        return out
    t = np.arange(marker_end - marker_start, dtype=np.float32) / float(sample_rate)
    beep = amplitude * np.sin(2.0 * math.pi * frequency_hz * t)
    fade = min(len(beep) // 2, max(1, int(round(0.01 * sample_rate))))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        beep[:fade] *= ramp
        beep[-fade:] *= ramp[::-1]
    out[marker_start:marker_end] += beep.astype(np.float32)
    return np.clip(out, -0.99, 0.99).astype(np.float32)


def rewrite_question(
    record: Mapping[str, Any],
    *,
    mode: str,
    answer_spans: Sequence[tuple[float, float]],
) -> str:
    if mode == "target_only":
        return (
            "Identify the sound in the isolated answer evidence. "
            "If the evidence is silent or contains no valid answer sound, choose no_evidence."
        )
    if mode == "answer_crop_front_norm":
        return (
            "A short isolated answer clip starts near the beginning of the audio. "
            "Which option names that sound? If the clip is silent, choose no_evidence."
        )
    if mode == "context_original":
        return str(record["question"])
    if mode == "context_span_text":
        if answer_spans:
            start = min(a for a, _ in answer_spans)
            end = max(b for _, b in answer_spans)
            return (
                f"The answer span is from {start:.2f} to {end:.2f} seconds in the audio. "
                "Which option names the sound in that span? "
                "If no valid answer span exists, choose no_evidence."
            )
        return (
            "No valid answer span exists for this question in the audio. "
            "Choose the correct option."
        )
    if mode == "context_marked":
        return (
            "A short beep marker ends immediately before the answer sound begins. "
            "Which option names the sound after the marker? "
            "If no marker or valid answer sound is present, choose no_evidence."
        )
    raise ValueError(f"unknown mode: {mode}")


def select_records(
    records: Sequence[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(records):
        return list(records)
    rng = random.Random(seed)
    answerable = [row for row in records if not bool(row.get("no_evidence"))]
    no_evidence = [row for row in records if bool(row.get("no_evidence"))]
    rng.shuffle(answerable)
    rng.shuffle(no_evidence)
    noev_count = min(len(no_evidence), max(1, round(sample_size * 0.27)))
    ans_count = sample_size - noev_count
    selected = answerable[:ans_count] + no_evidence[:noev_count]
    rng.shuffle(selected)
    return selected


def render_mode(
    *,
    mode: str,
    records: Sequence[dict[str, Any]],
    dataset_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mode_root = output_root / mode
    pred_root = mode_root / "predictions"
    manifest_path = mode_root / "manifest.jsonl"
    if mode_root.exists():
        shutil.rmtree(mode_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for record in records:
        sample_rate = int(record["sample_rate"])
        num_samples = int(record["num_samples"])
        mixture_path = safe_child(dataset_root, str(record["mixture_path"]))
        mixture, mixture_rate = read_mono(mixture_path)
        if mixture_rate != sample_rate:
            raise ValueError(f"sample-rate mismatch at {mixture_path}: {mixture_rate}")
        if mixture.shape[0] != num_samples:
            raise ValueError(f"sample-count mismatch at {mixture_path}: {mixture.shape[0]}")

        ans_spans = answer_intervals(record)
        ctx_spans = evidence_intervals(record)
        if mode in {"target_only", "answer_crop_front_norm"}:
            answer_stems = sum_stems(
                record,
                dataset_root,
                [str(x) for x in record.get("answer_event_ids", [])],
                num_samples=num_samples,
                sample_rate=sample_rate,
            )
            if mode == "target_only" or not ans_spans:
                evidence = answer_stems
            else:
                start = max(0.0, min(a for a, _ in ans_spans) - args.padding_seconds)
                end = min(
                    num_samples / sample_rate,
                    max(b for _, b in ans_spans) + args.padding_seconds,
                )
                src_a = max(0, min(num_samples, int(round(start * sample_rate))))
                src_b = max(0, min(num_samples, int(round(end * sample_rate))))
                evidence = np.zeros(num_samples, dtype=np.float32)
                if src_b > src_a:
                    clip = answer_stems[src_a:src_b].copy()
                    peak = float(np.max(np.abs(clip))) if clip.size else 0.0
                    if peak > 1e-5:
                        clip *= min(20.0, 0.70 / peak)
                    dst_a = int(round(0.25 * sample_rate))
                    dst_b = min(num_samples, dst_a + clip.shape[0])
                    evidence[dst_a:dst_b] = clip[: dst_b - dst_a]
        else:
            mask = context_mask(
                ctx_spans,
                num_samples=num_samples,
                sample_rate=sample_rate,
                padding_seconds=float(args.padding_seconds),
                fade_milliseconds=float(args.fade_milliseconds),
            )
            evidence = (mixture * mask).astype(np.float32)
            if mode == "context_marked":
                evidence = add_marker(
                    evidence,
                    answer_spans=ans_spans,
                    sample_rate=sample_rate,
                    frequency_hz=float(args.marker_frequency_hz),
                    duration_seconds=float(args.marker_duration_seconds),
                    amplitude=float(args.marker_amplitude),
                )
        residual = (mixture - evidence).astype(np.float32)

        qdir = pred_root / str(record["scene_id"]) / f"q{record['question_index']}_{record['question_type']}"
        qdir.mkdir(parents=True, exist_ok=True)
        sf.write(qdir / "predicted_evidence.wav", evidence, sample_rate)
        sf.write(qdir / "predicted_residual.wav", residual, sample_rate)
        (qdir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": record["id"],
                    "diagnostic_mode": mode,
                    "original_question": record["question"],
                    "diagnostic_question": rewrite_question(
                        record, mode=mode, answer_spans=ans_spans
                    ),
                    "answer": record["answer"],
                    "answer_event_ids": record.get("answer_event_ids", []),
                    "anchor_event_ids": record.get("anchor_event_ids", []),
                    "answer_intervals": ans_spans,
                    "context_intervals": ctx_spans,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        row = dict(record)
        row["question"] = rewrite_question(record, mode=mode, answer_spans=ans_spans)
        manifest_rows.append(row)
        item_rows.append(
            {
                "id": record["id"],
                "scene_id": record["scene_id"],
                "question_index": record["question_index"],
                "question_type": record["question_type"],
                "relation": record["relation"],
                "answer": record["answer"],
                "no_evidence": bool(record.get("no_evidence")),
                "diagnostic_mode": mode,
                "answer_intervals": ans_spans,
                "context_intervals": ctx_spans,
                "predicted_evidence_sha256": sha256_file(
                    qdir / "predicted_evidence.wav"
                ),
            }
        )

    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(mode_root / "items.jsonl", item_rows)
    schema_versions = sorted({str(row["schema_version"]) for row in manifest_rows})
    (pred_root / "evaluation_report.json").write_text(
        json.dumps(
            {
                "format": "qces_v5_temporal_only_baselines_v1",
                "schema_versions": schema_versions,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "items": [
                    {
                        "id": item["id"],
                        "scene_id": item["scene_id"],
                        "scene_family_id": next(
                            str(row["scene_family_id"])
                            for row in manifest_rows
                            if row["id"] == item["id"]
                        ),
                        "mode": mode,
                        "split": next(
                            str(row["split"]) for row in manifest_rows if row["id"] == item["id"]
                        ),
                    }
                    for item in item_rows
                ],
                "summaries": {
                    mode: {
                        "oracle_pointer_valid_records_↑": float(len(item_rows)),
                    }
                },
                "diagnostic_note": (
                    "Oracle pointer diagnostic audio; not a trainable separator "
                    "checkpoint and not paper-system predictions."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "mode": mode,
        "manifest": str(manifest_path),
        "predictions_root": str(pred_root),
        "num_records": len(manifest_rows),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = args.manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output root is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = select_records(read_jsonl(manifest), args.sample_size, args.seed)
    report = {
        "format": "qces_oracle_pointer_diagnostic_v1",
        "source_manifest": str(manifest),
        "dataset_root": str(dataset_root),
        "sample_size": len(records),
        "seed": args.seed,
        "modes": [],
    }
    for mode in args.modes:
        report["modes"].append(
            render_mode(
                mode=mode,
                records=records,
                dataset_root=dataset_root,
                output_root=output_root,
                args=args,
            )
        )
    (output_root / "render_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
