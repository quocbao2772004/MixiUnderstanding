#!/usr/bin/env python3
"""Render predicted pointer-style evidence audio from QCES RankCal receipts.

This is the deployable counterpart of the oracle pointer diagnostic:

* predicted_answer_crop_front_norm:
  infer the answer span from RankCal selected events, crop that span from the
  mixture, move it near the beginning of the clip, and normalize peak level.

* predicted_context_span_text:
  keep the RankCal selected-event context window from the mixture and rewrite
  the question to expose the predicted answer span times (not labels).

The script writes evaluator-compatible prediction roots plus strict QCES-v5
provenance reports.  It does not use oracle answer labels or oracle spans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


MODES = (
    "predicted_answer_crop_front_norm",
    "predicted_context_span_text",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rankcal-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--padding-seconds", type=float, default=0.20)
    parser.add_argument("--fade-milliseconds", type=float, default=10.0)
    parser.add_argument("--front-offset-seconds", type=float, default=0.25)
    parser.add_argument("--target-peak", type=float, default=0.70)
    parser.add_argument("--max-gain", type=float, default=20.0)
    parser.add_argument("--max-records", type=int)
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
        raise ValueError(f"unsafe path outside root: {path}")
    return path


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        waveform = waveform.mean(axis=1, keepdims=True)
    mono = np.ascontiguousarray(waveform[:, 0])
    if not np.isfinite(mono).all():
        raise ValueError(f"non-finite audio at {path}")
    return mono, int(sample_rate)


def load_rankcal(path: Path) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["id"])] = row
    return out


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


def selected_spans(rank_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = rank_row.get("selected_events", [])
    if not isinstance(events, list):
        return []
    spans = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        try:
            start = float(event["onset_seconds"])
            end = float(event["offset_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        spans.append(
            {
                "label": str(event.get("label")),
                "onset_seconds": start,
                "offset_seconds": end,
                "confidence": event.get("confidence"),
            }
        )
    return sorted(spans, key=lambda item: (item["onset_seconds"], item["offset_seconds"]))


def infer_predicted_answer_spans(
    record: Mapping[str, Any], rank_row: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Infer answer spans from selected events without oracle labels/spans.

    RankCal selected events usually contain anchor + target.  They are not
    explicitly typed, so we use relation-specific rules:

    * after: answer is the latest selected event.
    * before: answer is the earliest selected event.
    * first: answer is the earliest selected event matching predicted_answer,
      falling back to the earliest selected event.
    """

    if bool(rank_row.get("predicted_no_evidence")):
        return []
    spans = selected_spans(rank_row)
    if not spans:
        return []
    relation = str(record.get("relation") or rank_row.get("relation") or "")
    predicted_answer = rank_row.get("predicted_answer")
    if relation == "before":
        return [spans[0]]
    if relation == "after":
        return [spans[-1]]
    if relation == "first":
        if predicted_answer is not None:
            matches = [span for span in spans if span["label"] == str(predicted_answer)]
            if matches:
                return [matches[0]]
        return [spans[0]]
    # Conservative fallback: crop the most confident selected event.
    return [
        max(
            spans,
            key=lambda item: float(item["confidence"])
            if isinstance(item.get("confidence"), (int, float))
            else 0.0,
        )
    ]


def context_spans(rank_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return selected_spans(rank_row)


def context_mask(
    spans: Sequence[Mapping[str, Any]],
    *,
    num_samples: int,
    sample_rate: int,
    padding_seconds: float,
    fade_milliseconds: float,
) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=np.float32)
    if not spans:
        return mask
    start = max(0.0, min(float(s["onset_seconds"]) for s in spans) - padding_seconds)
    end = min(
        num_samples / sample_rate,
        max(float(s["offset_seconds"]) for s in spans) + padding_seconds,
    )
    a = max(0, min(num_samples, int(round(start * sample_rate))))
    b = max(0, min(num_samples, int(round(end * sample_rate))))
    if b > a:
        mask[a:b] = 1.0
    return apply_fade(mask, sample_rate, fade_milliseconds)


def render_front_crop(
    mixture: np.ndarray,
    spans: Sequence[Mapping[str, Any]],
    *,
    sample_rate: int,
    padding_seconds: float,
    front_offset_seconds: float,
    target_peak: float,
    max_gain: float,
) -> tuple[np.ndarray, dict[str, float | None]]:
    evidence = np.zeros_like(mixture, dtype=np.float32)
    if not spans:
        return evidence, {"crop_start_seconds": None, "crop_end_seconds": None, "gain": 0.0}
    start = max(0.0, min(float(s["onset_seconds"]) for s in spans) - padding_seconds)
    end = min(
        mixture.shape[0] / sample_rate,
        max(float(s["offset_seconds"]) for s in spans) + padding_seconds,
    )
    src_a = max(0, min(mixture.shape[0], int(round(start * sample_rate))))
    src_b = max(0, min(mixture.shape[0], int(round(end * sample_rate))))
    if src_b <= src_a:
        return evidence, {"crop_start_seconds": start, "crop_end_seconds": end, "gain": 0.0}
    clip = mixture[src_a:src_b].copy()
    peak = float(np.max(np.abs(clip))) if clip.size else 0.0
    gain = min(max_gain, target_peak / peak) if peak > 1e-5 else 0.0
    if gain > 0.0:
        clip *= gain
    dst_a = max(0, int(round(front_offset_seconds * sample_rate)))
    dst_b = min(evidence.shape[0], dst_a + clip.shape[0])
    if dst_b > dst_a:
        evidence[dst_a:dst_b] = clip[: dst_b - dst_a]
    return np.clip(evidence, -0.99, 0.99).astype(np.float32), {
        "crop_start_seconds": float(start),
        "crop_end_seconds": float(end),
        "gain": float(gain),
    }


def rewrite_question(
    *,
    mode: str,
    original_question: str,
    answer_spans: Sequence[Mapping[str, Any]],
) -> str:
    if mode == "predicted_answer_crop_front_norm":
        return (
            "A short predicted answer-evidence clip starts near the beginning of the audio. "
            "Which option names that sound? If the clip is silent or no valid evidence is "
            "present, choose no_evidence."
        )
    if mode == "predicted_context_span_text":
        if answer_spans:
            start = min(float(s["onset_seconds"]) for s in answer_spans)
            end = max(float(s["offset_seconds"]) for s in answer_spans)
            return (
                f"The predicted answer span is from {start:.2f} to {end:.2f} seconds "
                "in the audio. Which option names the sound in that span? If no valid "
                "answer span is present, choose no_evidence."
            )
        return (
            "No valid predicted answer span is present in the audio. "
            "Choose the correct option."
        )
    raise ValueError(f"unsupported mode: {mode}")


def render_mode(
    *,
    mode: str,
    records: Sequence[dict[str, Any]],
    rankcal: Mapping[str, Mapping[str, Any]],
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
        sample_id = str(record["id"])
        rank_row = rankcal.get(sample_id)
        if rank_row is None:
            raise ValueError(f"missing rankcal row for {sample_id}")
        sample_rate = int(record["sample_rate"])
        num_samples = int(record["num_samples"])
        mixture_path = safe_child(dataset_root, str(record["mixture_path"]))
        mixture, mixture_rate = read_mono(mixture_path)
        if mixture_rate != sample_rate:
            raise ValueError(f"sample-rate mismatch at {mixture_path}: {mixture_rate}")
        if mixture.shape[0] != num_samples:
            raise ValueError(f"sample-count mismatch at {mixture_path}: {mixture.shape[0]}")

        answer_spans = infer_predicted_answer_spans(record, rank_row)
        all_context_spans = context_spans(rank_row)
        if mode == "predicted_answer_crop_front_norm":
            evidence, crop_meta = render_front_crop(
                mixture,
                answer_spans,
                sample_rate=sample_rate,
                padding_seconds=float(args.padding_seconds),
                front_offset_seconds=float(args.front_offset_seconds),
                target_peak=float(args.target_peak),
                max_gain=float(args.max_gain),
            )
        elif mode == "predicted_context_span_text":
            mask = context_mask(
                all_context_spans,
                num_samples=num_samples,
                sample_rate=sample_rate,
                padding_seconds=float(args.padding_seconds),
                fade_milliseconds=float(args.fade_milliseconds),
            )
            evidence = (mixture * mask).astype(np.float32)
            crop_meta = {
                "crop_start_seconds": min(
                    (float(s["onset_seconds"]) for s in all_context_spans),
                    default=None,
                ),
                "crop_end_seconds": max(
                    (float(s["offset_seconds"]) for s in all_context_spans),
                    default=None,
                ),
                "gain": 1.0,
            }
        else:  # pragma: no cover
            raise ValueError(f"unsupported mode: {mode}")
        residual = (mixture - evidence).astype(np.float32)

        qdir = pred_root / str(record["scene_id"]) / f"q{record['question_index']}_{record['question_type']}"
        qdir.mkdir(parents=True, exist_ok=True)
        sf.write(qdir / "predicted_evidence.wav", evidence, sample_rate)
        sf.write(qdir / "predicted_residual.wav", residual, sample_rate)
        rewritten_question = rewrite_question(
            mode=mode,
            original_question=str(record["question"]),
            answer_spans=answer_spans,
        )
        (qdir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": sample_id,
                    "mode": mode,
                    "original_question": record["question"],
                    "rewritten_question": rewritten_question,
                    "gold_answer": record.get("answer"),
                    "gold_no_evidence": bool(record.get("no_evidence")),
                    "rankcal_predicted_answer": rank_row.get("predicted_answer"),
                    "rankcal_predicted_no_evidence": bool(
                        rank_row.get("predicted_no_evidence")
                    ),
                    "rankcal_score": rank_row.get("score"),
                    "rankcal_noev_probability": rank_row.get("noev_probability"),
                    "selected_events": all_context_spans,
                    "inferred_predicted_answer_spans": answer_spans,
                    "crop_meta": crop_meta,
                    "render_note": "no oracle answer label/span used",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        row = dict(record)
        row["question"] = rewritten_question
        manifest_rows.append(row)
        item_rows.append(
            {
                "id": sample_id,
                "scene_id": record["scene_id"],
                "scene_family_id": record["scene_family_id"],
                "split": record["split"],
                "question_index": record["question_index"],
                "question_type": record["question_type"],
                "relation": record["relation"],
                "answer": record.get("answer"),
                "no_evidence": bool(record.get("no_evidence")),
                "mode": mode,
                "rankcal_predicted_answer": rank_row.get("predicted_answer"),
                "rankcal_predicted_no_evidence": bool(
                    rank_row.get("predicted_no_evidence")
                ),
                "rankcal_answer_correct": rank_row.get("answer_correct"),
                "rankcal_score_↑": rank_row.get("score"),
                "rankcal_noev_probability_↓": rank_row.get("noev_probability"),
                "selected_labels": [span["label"] for span in all_context_spans],
                "inferred_answer_spans": answer_spans,
                "crop_meta": crop_meta,
                "predicted_evidence_sha256": sha256_file(qdir / "predicted_evidence.wav"),
            }
        )

    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(mode_root / "items.jsonl", item_rows)
    schema_versions = sorted({str(row["schema_version"]) for row in manifest_rows})
    summary_answer = [
        item
        for item in item_rows
        if not item["no_evidence"] and item["rankcal_answer_correct"] is not None
    ]
    summary_noev = [item for item in item_rows if item["no_evidence"]]
    summary = {
        "rendered_records_↑": float(len(item_rows)),
        "rankcal_answer_accuracy_on_rendered_answerable_↑": (
            float(sum(bool(item["rankcal_answer_correct"]) for item in summary_answer))
            / len(summary_answer)
            if summary_answer
            else 0.0
        ),
        "rankcal_no_evidence_accuracy_on_rendered_noev_↑": (
            float(
                sum(bool(item["rankcal_predicted_no_evidence"]) for item in summary_noev)
            )
            / len(summary_noev)
            if summary_noev
            else 0.0
        ),
    }
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
                        "scene_family_id": item["scene_family_id"],
                        "mode": mode,
                        "split": item["split"],
                    }
                    for item in item_rows
                ],
                "summaries": {mode: summary},
                "diagnostic_note": (
                    "Predicted pointer evidence generated from RankCal selected events; "
                    "no oracle answer labels or oracle spans are used."
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
        "num_records": len(item_rows),
        "summary": summary,
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

    records = read_jsonl(manifest)
    if args.max_records is not None:
        records = records[: int(args.max_records)]
    rankcal = load_rankcal(args.rankcal_jsonl.resolve())
    report = {
        "format": "qces_predicted_pointer_audio_v1",
        "source_manifest": str(manifest),
        "dataset_root": str(dataset_root),
        "rankcal_jsonl": str(args.rankcal_jsonl.resolve()),
        "num_records": len(records),
        "modes": [],
    }
    for mode in args.modes:
        report["modes"].append(
            render_mode(
                mode=mode,
                records=records,
                rankcal=rankcal,
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
