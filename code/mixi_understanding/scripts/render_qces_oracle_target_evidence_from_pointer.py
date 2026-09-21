#!/usr/bin/env python3
"""Render ablation-C oracle target-evidence spans for QCES pointer prompts.

This is a diagnostic artifact, not a deployable system.  It keeps the input
manifest exactly as produced by the pointer/span-text pipeline, including the
question text derived from the model prediction.  Only the audio evidence is
changed: ``predicted_evidence.wav`` becomes the original mixture masked by the
gold target-evidence time span from ``evidence_event_ids``.

Use this to test whether improving temporal IoU alone would help downstream
AudioQA, without giving the evaluator oracle answer labels or clean oracle
stems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


PREDICTED_PROMPT_MODE = "oracle_target_evidence_span__predicted_prompt"
ORACLE_ANSWER_SPAN_PROMPT_MODE = (
    "oracle_target_evidence_span__oracle_answer_span_prompt"
)
ORACLE_CLEAN_STEM_PROMPT_MODE = "oracle_clean_stem__oracle_answer_span_prompt"
ORACLE_ANSWER_CLEAN_STEM_PROMPT_MODE = (
    "oracle_answer_clean_stem__oracle_answer_span_prompt"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("predicted_prompt", "oracle_answer_span_prompt"),
        default="predicted_prompt",
        help=(
            "predicted_prompt keeps the question text from the input manifest. "
            "oracle_answer_span_prompt rewrites it with the gold answer span; "
            "use this for the clean temporal-IoU upper-bound ablation."
        ),
    )
    parser.add_argument(
        "--audio-mode",
        choices=("mixture_mask", "clean_stem", "answer_clean_stem"),
        default="mixture_mask",
        help=(
            "mixture_mask keeps all mixture audio inside gold evidence spans. "
            "clean_stem sums only the gold evidence_event_ids stems. "
            "answer_clean_stem sums only the gold answer_event_ids stems."
        ),
    )
    parser.add_argument("--fade-milliseconds", type=float, default=10.0)
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


def sum_event_stems(
    record: Mapping[str, Any],
    dataset_root: Path,
    event_ids: Sequence[str],
    *,
    num_samples: int,
    sample_rate: int,
) -> np.ndarray:
    events = event_map(record)
    out = np.zeros(num_samples, dtype=np.float32)
    for event_id in event_ids:
        event = events.get(str(event_id))
        if event is None:
            continue
        stem_path = safe_child(dataset_root, str(event["stem_path"]))
        stem, stem_rate = read_mono(stem_path)
        if stem_rate != sample_rate:
            raise ValueError(f"sample-rate mismatch at {stem_path}: {stem_rate}")
        if stem.shape[0] != num_samples:
            raise ValueError(f"sample-count mismatch at {stem_path}: {stem.shape[0]}")
        out += stem
    return out.astype(np.float32)


def event_intervals(record: Mapping[str, Any]) -> list[tuple[float, float]]:
    wanted = {str(event_id) for event_id in record.get("evidence_event_ids", [])}
    spans: list[tuple[float, float]] = []
    for event in record.get("events", []):
        if str(event.get("event_id")) not in wanted:
            continue
        start = float(event["onset_seconds"])
        end = float(event["offset_seconds"])
        if end > start:
            spans.append((start, end))
    return merge_intervals(spans)


def answer_intervals(record: Mapping[str, Any]) -> list[tuple[float, float]]:
    spans = []
    value = record.get("answer_intervals")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list) and len(item) == 2:
                start = float(item[0])
                end = float(item[1])
                if end > start:
                    spans.append((start, end))
    if spans:
        return merge_intervals(spans)

    wanted = {str(event_id) for event_id in record.get("answer_event_ids", [])}
    for event in record.get("events", []):
        if str(event.get("event_id")) not in wanted:
            continue
        start = float(event["onset_seconds"])
        end = float(event["offset_seconds"])
        if end > start:
            spans.append((start, end))
    return merge_intervals(spans)


def rewrite_oracle_answer_span_question(record: Mapping[str, Any]) -> str:
    spans = answer_intervals(record)
    if spans:
        start = min(a for a, _ in spans)
        end = max(b for _, b in spans)
        return (
            f"The answer span is from {start:.2f} to {end:.2f} seconds in the audio. "
            "Which option names the sound in that span? "
            "If no valid answer span exists, choose no_evidence."
        )
    return (
        "No valid answer span exists in the audio for this question. "
        "Choose no_evidence."
    )


def merge_intervals(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


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


def span_mask(
    spans: Sequence[tuple[float, float]],
    *,
    num_samples: int,
    sample_rate: int,
    fade_milliseconds: float,
) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=np.float32)
    for start, end in spans:
        a = max(0, min(num_samples, int(round(start * sample_rate))))
        b = max(0, min(num_samples, int(round(end * sample_rate))))
        if b > a:
            mask[a:b] = 1.0
    return apply_fade(mask, sample_rate, fade_milliseconds)


def render(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if args.audio_mode == "answer_clean_stem":
        if args.prompt_mode != "oracle_answer_span_prompt":
            raise SystemExit(
                "answer_clean_stem audio requires --prompt-mode oracle_answer_span_prompt"
            )
        mode = ORACLE_ANSWER_CLEAN_STEM_PROMPT_MODE
    elif args.audio_mode == "clean_stem":
        if args.prompt_mode != "oracle_answer_span_prompt":
            raise SystemExit("clean_stem audio requires --prompt-mode oracle_answer_span_prompt")
        mode = ORACLE_CLEAN_STEM_PROMPT_MODE
    elif args.prompt_mode == "oracle_answer_span_prompt":
        mode = ORACLE_ANSWER_SPAN_PROMPT_MODE
    else:
        mode = PREDICTED_PROMPT_MODE
    mode_root = output_root / mode
    pred_root = mode_root / "predictions"
    if mode_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {mode_root}")
        shutil.rmtree(mode_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(manifest)
    manifest_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    total_target_duration = 0.0
    answerable = 0

    for record in records:
        sample_id = str(record["id"])
        sample_rate = int(record["sample_rate"])
        num_samples = int(record["num_samples"])
        mixture_path = safe_child(dataset_root, str(record["mixture_path"]))
        mixture, mixture_rate = read_mono(mixture_path)
        if mixture_rate != sample_rate:
            raise ValueError(f"sample-rate mismatch at {mixture_path}: {mixture_rate}")
        if mixture.shape[0] != num_samples:
            raise ValueError(
                f"sample-count mismatch at {mixture_path}: {mixture.shape[0]}"
            )

        spans = event_intervals(record)
        total_target_duration += sum(end - start for start, end in spans)
        if not bool(record.get("no_evidence")):
            answerable += 1
        if args.audio_mode == "answer_clean_stem":
            evidence = sum_event_stems(
                record,
                dataset_root,
                [str(x) for x in record.get("answer_event_ids", [])],
                num_samples=num_samples,
                sample_rate=sample_rate,
            )
        elif args.audio_mode == "clean_stem":
            evidence = sum_event_stems(
                record,
                dataset_root,
                [str(x) for x in record.get("evidence_event_ids", [])],
                num_samples=num_samples,
                sample_rate=sample_rate,
            )
        else:
            mask = span_mask(
                spans,
                num_samples=num_samples,
                sample_rate=sample_rate,
                fade_milliseconds=float(args.fade_milliseconds),
            )
            evidence = (mixture * mask).astype(np.float32)
        residual = (mixture - evidence).astype(np.float32)

        qdir = (
            pred_root
            / str(record["scene_id"])
            / f"q{record['question_index']}_{record['question_type']}"
        )
        qdir.mkdir(parents=True, exist_ok=True)
        sf.write(qdir / "predicted_evidence.wav", evidence, sample_rate)
        sf.write(qdir / "predicted_residual.wav", residual, sample_rate)
        (qdir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": sample_id,
                    "mode": mode,
                    "prompt_mode": args.prompt_mode,
                    "audio_mode": args.audio_mode,
                    "question_kept_from_manifest": record.get("question"),
                    "rendered_question": (
                        rewrite_oracle_answer_span_question(record)
                        if args.prompt_mode == "oracle_answer_span_prompt"
                        else record.get("question")
                    ),
                    "gold_answer": record.get("answer"),
                    "gold_no_evidence": bool(record.get("no_evidence")),
                    "evidence_event_ids": record.get("evidence_event_ids", []),
                    "answer_intervals": answer_intervals(record),
                    "oracle_target_evidence_intervals": spans,
                    "render_note": (
                        "Ablation C: oracle target-evidence time mask on mixture; "
                        "prompt mode is recorded separately."
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_row = dict(record)
        if args.prompt_mode == "oracle_answer_span_prompt":
            manifest_row["question"] = rewrite_oracle_answer_span_question(record)
        manifest_rows.append(manifest_row)
        item_rows.append(
            {
                "id": sample_id,
                "scene_id": record["scene_id"],
                "scene_family_id": record["scene_family_id"],
                "split": record["split"],
                "mode": mode,
                "prompt_mode": args.prompt_mode,
                "audio_mode": args.audio_mode,
                "question_index": record["question_index"],
                "question_type": record["question_type"],
                "relation": record["relation"],
                "answer": record.get("answer"),
                "no_evidence": bool(record.get("no_evidence")),
                "answer_intervals": answer_intervals(record),
                "oracle_target_evidence_intervals": spans,
                "predicted_evidence_sha256": sha256_file(
                    qdir / "predicted_evidence.wav"
                ),
            }
        )

    write_jsonl(mode_root / "manifest.jsonl", manifest_rows)
    write_jsonl(mode_root / "items.jsonl", item_rows)
    schema_versions = sorted({str(row.get("schema_version")) for row in manifest_rows})
    report = {
        "format": "qces_v5_temporal_only_baselines_v1",
        "mode": mode,
        "prompt_mode": args.prompt_mode,
        "audio_mode": args.audio_mode,
        "source_manifest": str(manifest),
        "manifest": str((mode_root / "manifest.jsonl").resolve()),
        "manifest_sha256": sha256_file(mode_root / "manifest.jsonl"),
        "schema_versions": schema_versions,
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
        "summaries": {
            mode: {
                "rendered_records_↑": float(len(item_rows)),
                "answerable_records_↑": float(answerable),
                "no_evidence_records_↓": float(len(item_rows) - answerable),
                "mean_oracle_target_evidence_duration_seconds_↓": (
                    total_target_duration / len(item_rows) if item_rows else 0.0
                ),
            }
        },
        "diagnostic_note": (
            "Oracle target-evidence diagnostic. With audio_mode=mixture_mask, "
            "audio is mixture masked by gold evidence_event_ids spans. With "
            "audio_mode=clean_stem, audio is the sum of clean gold "
            "evidence_event_ids stems. With audio_mode=answer_clean_stem, audio is "
            "the sum of clean gold answer_event_ids stems. With prompt_mode=oracle_answer_span_prompt, "
            "the manifest question is rewritten using the gold answer span."
        ),
    }
    (pred_root / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "records": len(item_rows),
        "answerable": answerable,
        "no_evidence": len(item_rows) - answerable,
        "mode_root": str(mode_root),
        "predictions_root": str(pred_root),
        "report": str(pred_root / "evaluation_report.json"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(render(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
