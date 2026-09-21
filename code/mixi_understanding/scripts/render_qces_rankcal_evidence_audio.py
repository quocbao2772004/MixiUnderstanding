#!/usr/bin/env python3
"""Render deployable evidence WAVs from QCES-RankCal selected event spans.

This script creates an evaluator-compatible prediction root, but it does not
use oracle stems.  Evidence is produced by masking/cropping the mixture around
the events selected by a RankCal JSONL receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qces_v5_schema import parse_qces_v5_record
from mixi_understanding.qces.data import read_jsonl

FORMAT = "qces_v6_pipeline_evaluation_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rankcal-jsonl", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("rankcal_span_crop", "rankcal_context_window"),
        required=True,
    )
    parser.add_argument("--padding-seconds", type=float, default=0.20)
    parser.add_argument("--fade-milliseconds", type=float, default=10.0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_child(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"unsafe child path: {path}")
    return path


def load_records(path: Path) -> list[Any]:
    return [parse_qces_v5_record(row) for row in read_jsonl(path)]


def load_rankcal(path: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            result[item["id"]] = item
    return result


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


def spans_to_mask(
    *,
    spans: Sequence[tuple[float, float]],
    sample_rate: int,
    num_samples: int,
    padding_seconds: float,
    mode: str,
    fade_milliseconds: float,
) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=np.float32)
    if not spans:
        return mask
    padded = [
        (
            max(0.0, float(start) - padding_seconds),
            min(num_samples / sample_rate, float(end) + padding_seconds),
        )
        for start, end in spans
    ]
    if mode == "rankcal_context_window":
        padded = [(min(start for start, _ in padded), max(end for _, end in padded))]
    for start, end in padded:
        a = max(0, min(num_samples, int(round(start * sample_rate))))
        b = max(0, min(num_samples, int(round(end * sample_rate))))
        if b > a:
            mask[a:b] = 1.0
    return apply_fade(mask, sample_rate, fade_milliseconds)


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

    records = load_records(manifest)
    if args.max_records is not None:
        if args.max_records <= 0:
            raise SystemExit("--max-records must be positive")
        records = records[: args.max_records]
    rankcal = load_rankcal(args.rankcal_jsonl.resolve())
    base_report = json.loads(args.base_report.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    answer_correct = answer_total = noev_correct = noev_total = 0
    span_iou_sum = span_iou_count = 0

    for record in records:
        if record.sample_id not in rankcal:
            raise ValueError(f"missing rankcal record: {record.sample_id}")
        pred = rankcal[record.sample_id]
        evidence_events = pred.get("selected_events", [])
        if not isinstance(evidence_events, list):
            raise ValueError("selected_events must be a list")
        spans = [
            (float(event["onset_seconds"]), float(event["offset_seconds"]))
            for event in evidence_events
            if isinstance(event, Mapping)
        ]
        mixture_path = safe_child(dataset_root, record.mixture_path)
        waveform, sample_rate = sf.read(mixture_path, dtype="float32", always_2d=False)
        if sample_rate != record.sample_rate:
            raise ValueError(f"sample-rate mismatch for {record.sample_id}")
        mono = waveform if waveform.ndim == 1 else waveform.mean(axis=1)
        if mono.shape[0] != record.num_samples:
            raise ValueError(f"sample-count mismatch for {record.sample_id}")
        if bool(pred.get("predicted_no_evidence")):
            mask = np.zeros(record.num_samples, dtype=np.float32)
        else:
            mask = spans_to_mask(
                spans=spans,
                sample_rate=record.sample_rate,
                num_samples=record.num_samples,
                padding_seconds=args.padding_seconds,
                mode=args.mode,
                fade_milliseconds=args.fade_milliseconds,
            )
        evidence = (mono * mask).astype(np.float32)
        residual = (mono - evidence).astype(np.float32)
        qdir = output_root / record.scene_id / f"q{record.question_index}_{record.question_type}"
        qdir.mkdir(parents=True, exist_ok=True)
        sf.write(qdir / "predicted_evidence.wav", evidence, record.sample_rate)
        sf.write(qdir / "predicted_residual.wav", residual, record.sample_rate)
        (qdir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": record.sample_id,
                    "mode": args.mode,
                    "question": record.question,
                    "rankcal_predicted_answer": pred.get("predicted_answer"),
                    "rankcal_predicted_no_evidence": pred.get("predicted_no_evidence"),
                    "selected_events": evidence_events,
                    "render_recipe": {
                        "source": "mixture_mask_from_rankcal_selected_event_spans",
                        "padding_seconds": args.padding_seconds,
                        "fade_milliseconds": args.fade_milliseconds,
                        "context_window": args.mode == "rankcal_context_window",
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if record.no_evidence:
            noev_total += 1
            noev_correct += int(bool(pred.get("predicted_no_evidence")))
        else:
            answer_total += 1
            answer_correct += int(pred.get("predicted_answer") == record.answer)
            iou = pred.get("span_iou_↑")
            if isinstance(iou, (int, float)):
                span_iou_sum += float(iou)
                span_iou_count += 1
        items.append(
            {
                "id": record.sample_id,
                "scene_id": record.scene_id,
                "scene_family_id": record.scene_family_id,
                "split": record.split,
                "evaluation_axis": record.evaluation_axis,
                "variant_id": record.variant_id,
                "question": record.question,
                "question_index": record.question_index,
                "question_type": record.question_type,
                "relation": record.relation,
                "answer": record.answer,
                "no_evidence": bool(record.no_evidence),
                "mode": args.mode,
                "planned_no_evidence": bool(pred.get("predicted_no_evidence")),
                "planned_answer_label": pred.get("predicted_answer"),
                "planned_labels": [event.get("label") for event in evidence_events],
                "planned_reason": "rankcal_selected_events_to_mixture_mask",
                "planner_answer_correct": (
                    None
                    if record.no_evidence
                    else pred.get("predicted_answer") == record.answer
                ),
                "planner_no_evidence_correct": (
                    bool(pred.get("predicted_no_evidence")) == bool(record.no_evidence)
                ),
                "planner_span_iou_↑": pred.get("span_iou_↑"),
                "predicted_spans": spans,
                "descriptives": {
                    "evidence_retained_ratio": float(np.mean(mask > 0.0)),
                    "evidence_absolute_peak": float(np.max(np.abs(evidence)))
                    if evidence.size
                    else 0.0,
                },
                "metrics": {
                    "mixture_consistency_l1_sanity_↓": 0.0,
                    "rankcal_score_↑": pred.get("score"),
                    "rankcal_noev_probability_↓": pred.get("noev_probability"),
                },
            }
        )

    summary = {
        "record_count": len(records),
        "answerable_count": answer_total,
        "no_evidence_count": noev_total,
        "planner_answer_accuracy_↑": answer_correct / answer_total,
        "planner_no_evidence_accuracy_↑": noev_correct / noev_total,
        "planner_span_iou_mean_↑": (
            span_iou_sum / span_iou_count if span_iou_count else None
        ),
        "mixture_consistency_l1_sanity_maximum_↓": 0.0,
    }
    report = {
        "format": FORMAT,
        "schema_versions": base_report["schema_versions"],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "proposal_head": base_report.get("proposal_head"),
        "rankcal_jsonl": {
            "path": str(args.rankcal_jsonl.resolve()),
            "sha256": sha256_file(args.rankcal_jsonl.resolve()),
        },
        "render_mode": args.mode,
        "items": items,
        "summaries_by_mode": {args.mode: summary},
    }
    (output_root / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_root}")


if __name__ == "__main__":
    main()
