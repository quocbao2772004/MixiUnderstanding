#!/usr/bin/env python3
"""Render answer-only predicted evidence audio for QCES pointer experiments.

This is a separate, non-invasive variant of ``render_qces_predicted_pointer_audio``.
It uses the existing RankCal/pointer output, but renders only the inferred answer
event span instead of the whole anchor+answer context window.

No oracle answer labels, oracle spans, or oracle stems are used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


MODE = "predicted_answer_only_span_text"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rankcal-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--padding-seconds", type=float, default=0.10)
    parser.add_argument("--fade-milliseconds", type=float, default=10.0)
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
    rows: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def selected_spans(rank_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = rank_row.get("selected_events", [])
    if not isinstance(events, list):
        return []
    spans: list[dict[str, Any]] = []
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
    """Infer the answer event from RankCal selected events.

    The RankCal row only provides selected events, not explicit event roles.  We
    use the same deployable rules as the existing pointer renderer:

    * ``before``: answer is the earliest selected event.
    * ``after``: answer is the latest selected event.
    * ``first``: earliest selected event matching the predicted answer label,
      otherwise earliest selected event.
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
    return [
        max(
            spans,
            key=lambda item: float(item["confidence"])
            if isinstance(item.get("confidence"), (int, float))
            else 0.0,
        )
    ]


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
            out[fall_start:end] = np.minimum(out[fall_start:end], ramp[:fall_len][::-1])
    return out


def span_mask(
    spans: Sequence[Mapping[str, Any]],
    *,
    num_samples: int,
    sample_rate: int,
    padding_seconds: float,
    fade_milliseconds: float,
) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=np.float32)
    for span in spans:
        start = max(0.0, float(span["onset_seconds"]) - padding_seconds)
        end = min(num_samples / sample_rate, float(span["offset_seconds"]) + padding_seconds)
        a = max(0, min(num_samples, int(round(start * sample_rate))))
        b = max(0, min(num_samples, int(round(end * sample_rate))))
        if b > a:
            mask[a:b] = 1.0
    return apply_fade(mask, sample_rate, fade_milliseconds)


def rewrite_question(answer_spans: Sequence[Mapping[str, Any]]) -> str:
    if not answer_spans:
        return (
            "No valid predicted answer span is present in the audio. "
            "Choose no_evidence."
        )
    start = min(float(span["onset_seconds"]) for span in answer_spans)
    end = max(float(span["offset_seconds"]) for span in answer_spans)
    return (
        f"The audio contains only the predicted answer evidence from {start:.2f} "
        f"to {end:.2f} seconds of the original mixture. Which option names this "
        "sound? If no valid evidence is present, choose no_evidence."
    )


def render(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    mode_root = output_root / MODE
    pred_root = mode_root / "predictions"
    if mode_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output mode root exists: {mode_root}")
        shutil.rmtree(mode_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(manifest)
    if args.max_records is not None:
        records = records[: int(args.max_records)]
    rankcal = load_rankcal(args.rankcal_jsonl.resolve())

    manifest_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    total_evidence_seconds = 0.0
    total_active_records = 0

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
        mask = span_mask(
            answer_spans,
            num_samples=num_samples,
            sample_rate=sample_rate,
            padding_seconds=float(args.padding_seconds),
            fade_milliseconds=float(args.fade_milliseconds),
        )
        evidence = (mixture * mask).astype(np.float32)
        residual = (mixture - evidence).astype(np.float32)
        evidence_seconds = float(mask.mean() * num_samples / sample_rate)
        if evidence_seconds > 0.0:
            total_active_records += 1
            total_evidence_seconds += evidence_seconds

        qdir = pred_root / str(record["scene_id"]) / f"q{record['question_index']}_{record['question_type']}"
        qdir.mkdir(parents=True, exist_ok=True)
        sf.write(qdir / "predicted_evidence.wav", evidence, sample_rate)
        sf.write(qdir / "predicted_residual.wav", residual, sample_rate)

        rewritten_question = rewrite_question(answer_spans)
        (qdir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": sample_id,
                    "mode": MODE,
                    "original_question": record["question"],
                    "rewritten_question": rewritten_question,
                    "gold_answer": record.get("answer"),
                    "gold_no_evidence": bool(record.get("no_evidence")),
                    "rankcal_predicted_answer": rank_row.get("predicted_answer"),
                    "rankcal_predicted_no_evidence": bool(rank_row.get("predicted_no_evidence")),
                    "rankcal_score": rank_row.get("score"),
                    "rankcal_noev_probability": rank_row.get("noev_probability"),
                    "selected_events": selected_spans(rank_row),
                    "inferred_predicted_answer_spans": answer_spans,
                    "evidence_seconds": evidence_seconds,
                    "render_note": "predicted answer-only span; no oracle label/span/stem used",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        out_record = dict(record)
        out_record["question"] = rewritten_question
        manifest_rows.append(out_record)
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
                "mode": MODE,
                "rankcal_predicted_answer": rank_row.get("predicted_answer"),
                "rankcal_predicted_no_evidence": bool(rank_row.get("predicted_no_evidence")),
                "rankcal_answer_correct": rank_row.get("answer_correct"),
                "rankcal_score_↑": rank_row.get("score"),
                "rankcal_noev_probability_↓": rank_row.get("noev_probability"),
                "selected_labels": [span["label"] for span in selected_spans(rank_row)],
                "inferred_answer_spans": answer_spans,
                "evidence_seconds": evidence_seconds,
                "predicted_evidence_sha256": sha256_file(qdir / "predicted_evidence.wav"),
            }
        )

    manifest_path = mode_root / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(mode_root / "items.jsonl", item_rows)

    answerable = [item for item in item_rows if not item["no_evidence"]]
    noev = [item for item in item_rows if item["no_evidence"]]
    summary = {
        "rendered_records_↑": float(len(item_rows)),
        "answerable_records_↑": float(len(answerable)),
        "no_evidence_records_↑": float(len(noev)),
        "active_rendered_records_↑": float(total_active_records),
        "mean_nonempty_evidence_seconds_↓": (
            total_evidence_seconds / total_active_records if total_active_records else 0.0
        ),
        "rankcal_answer_accuracy_on_rendered_answerable_↑": (
            sum(bool(item["rankcal_answer_correct"]) for item in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        "rankcal_no_evidence_accuracy_on_rendered_noev_↑": (
            sum(bool(item["rankcal_predicted_no_evidence"]) for item in noev) / len(noev)
            if noev
            else 0.0
        ),
    }
    report = {
        "format": "qces_predicted_answer_only_audio_v1",
        "mode": MODE,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "dataset_root": str(dataset_root),
        "rankcal_jsonl": str(args.rankcal_jsonl.resolve()),
        "rankcal_jsonl_sha256": sha256_file(args.rankcal_jsonl.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_root": str(pred_root.resolve()),
        "num_records": len(item_rows),
        "summary": summary,
        "diagnostic_note": (
            "This predicted answer-only evidence is rendered from existing RankCal "
            "selected events and relation rules. It uses no oracle answer labels, "
            "oracle spans, or oracle stems."
        ),
    }
    (output_root / "render_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (pred_root / "evaluation_report.json").write_text(
        json.dumps(
            {
                "format": "qces_v5_temporal_only_baselines_v1",
                "schema_versions": sorted({str(row["schema_version"]) for row in manifest_rows}),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "items": [
                    {
                        "id": item["id"],
                        "scene_id": item["scene_id"],
                        "scene_family_id": item["scene_family_id"],
                        "mode": MODE,
                        "split": item["split"],
                    }
                    for item in item_rows
                ],
                "summaries": {MODE: summary},
                "diagnostic_note": report["diagnostic_note"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(render(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
