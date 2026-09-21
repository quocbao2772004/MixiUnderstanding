#!/usr/bin/env python3
"""One-page Streamlit demo for QCES predicted pointer evidence.

This app is intentionally read-only.  It visualizes the current predicted
context-span evidence experiment:

* original mixture audio,
* predicted context evidence and residual,
* original question, span-text prompt, answer options,
* RankCal selected events and inferred answer span,
* AF3 outputs when available.

It does not import the older demo contract so it can run even when historical
demo utilities change.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


DEFAULT_ORIGINAL_MANIFEST = (
    PROJECT_ROOT / "outputs/qces_v6_external_audioqa/fast384/qces_val_stratified384.jsonl"
)
DEFAULT_POINTER_ROOT = (
    PROJECT_ROOT
    / "outputs/qces_v6_predicted_pointer_audio/fast384_context_span_text/predicted_context_span_text"
)
DEFAULT_AF3_ROOT = (
    PROJECT_ROOT / "outputs/qces_v6_external_audioqa/fast384/af3_context_span_text"
)
DEFAULT_QWEN_ROOT = (
    PROJECT_ROOT / "outputs/qces_v6_external_audioqa/fast384/qwen2_audio_context_span_text"
)
DEFAULT_PHI_ROOT = (
    PROJECT_ROOT / "outputs/qces_v6_external_audioqa/fast384/phi4mm_context_span_text"
)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/qces_v6_full_cropbank_v2"
DEFAULT_RANKCAL_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v6_guarded_rankcal_results/guarded_v1/guarded_rankcal_report.json"
)
DEFAULT_VAL_EVIDENCE_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v6_predicted_pointer_audio/val_full/predicted_context_span_text"
    / "predictions/evaluation_report.json"
)

SPLIT_ORDER = [
    "train",
    "val",
    "test_iid",
    "test_compositional_ood",
    "test_label_ood",
]
SPLIT_LABELS = {
    "train": "train",
    "val": "val",
    "test_iid": "test-IID",
    "test_compositional_ood": "test-comp-OOD",
    "test_label_ood": "test-label-OOD",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--original-manifest", type=Path, default=DEFAULT_ORIGINAL_MANIFEST)
    parser.add_argument("--pointer-root", type=Path, default=DEFAULT_POINTER_ROOT)
    parser.add_argument("--af3-root", type=Path, default=DEFAULT_AF3_ROOT)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--phi-root", type=Path, default=DEFAULT_PHI_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--rankcal-report", type=Path, default=DEFAULT_RANKCAL_REPORT)
    parser.add_argument(
        "--val-evidence-report", type=Path, default=DEFAULT_VAL_EVIDENCE_REPORT
    )
    args, _ = parser.parse_known_args(argv)
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def safe_path(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"unsafe path: {path}")
    return path


def label_text(label: str | None) -> str:
    if label is None:
        return "—"
    return str(label).replace("_", " ")


def fmt_span(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "—"
    return f"{start:.2f}–{end:.2f}s"


def load_af3_items(path: Path) -> dict[str, dict[str, Mapping[str, Any]]]:
    by_id: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        by_id[str(row["id"])][str(row["condition"])] = row
    return dict(by_id)


def artifact_fingerprint(
    original_manifest: Path,
    pointer_root: Path,
    af3_root: Path,
    qwen_root: Path,
    phi_root: Path,
) -> str:
    paths = [
        original_manifest,
        pointer_root / "items.jsonl",
        pointer_root / "manifest.jsonl",
        af3_root / "mixture_original_question" / "items.jsonl",
        af3_root / "span_text_mixture_and_predicted_evidence" / "items.jsonl",
        qwen_root / "mixture_original_question" / "items.jsonl",
        qwen_root / "span_text_mixture_and_predicted_evidence" / "items.jsonl",
        phi_root / "mixture_original_question" / "items.jsonl",
        phi_root / "span_text_mixture_and_predicted_evidence" / "items.jsonl",
    ]
    parts = []
    for path in paths:
        if path.is_file():
            stat = path.stat()
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        else:
            parts.append(f"{path}:missing")
    return "|".join(parts)


def files_fingerprint(paths: Sequence[Path]) -> str:
    parts = []
    for path in paths:
        if path.is_file():
            stat = path.stat()
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        else:
            parts.append(f"{path}:missing")
    return "|".join(parts)


def dataset_fingerprint(dataset_root: Path) -> str:
    paths = [
        dataset_root / "dataset_config.json",
        dataset_root / "validation_report.json",
    ] + [dataset_root / f"qces_{split}.jsonl" for split in SPLIT_ORDER]
    return files_fingerprint(paths)


@st.cache_data(show_spinner=False)
def load_bundle(
    original_manifest: str,
    pointer_root: str,
    af3_root: str,
    qwen_root: str,
    phi_root: str,
    dataset_root: str,
    artifact_state: str,
) -> dict[str, Any]:
    _ = artifact_state
    original_path = Path(original_manifest)
    pointer = Path(pointer_root)
    af3 = Path(af3_root)
    qwen = Path(qwen_root)
    phi = Path(phi_root)
    dataset = Path(dataset_root)

    records = {str(row["id"]): row for row in read_jsonl(original_path)}
    pointer_items = {str(row["id"]): row for row in read_jsonl(pointer / "items.jsonl")}
    pointer_manifest = {str(row["id"]): row for row in read_jsonl(pointer / "manifest.jsonl")}
    mixture_original = load_af3_items(af3 / "mixture_original_question" / "items.jsonl")
    span_eval = load_af3_items(
        af3 / "span_text_mixture_and_predicted_evidence" / "items.jsonl"
    )
    qwen_mixture_original = load_af3_items(
        qwen / "mixture_original_question" / "items.jsonl"
    )
    qwen_span_eval = load_af3_items(
        qwen / "span_text_mixture_and_predicted_evidence" / "items.jsonl"
    )
    phi_mixture_original = load_af3_items(
        phi / "mixture_original_question" / "items.jsonl"
    )
    phi_span_eval = load_af3_items(
        phi / "span_text_mixture_and_predicted_evidence" / "items.jsonl"
    )

    case_ids = sorted(set(records) | set(pointer_items) | set(pointer_manifest))
    cases: list[dict[str, Any]] = []
    for sample_id in case_ids:
        record = records.get(sample_id) or pointer_manifest.get(sample_id)
        if not record:
            continue
        pointer_item = pointer_items.get(sample_id, {})
        scene_id = str(record["scene_id"])
        qdir = (
            pointer
            / "predictions"
            / scene_id
            / f"q{record['question_index']}_{record['question_type']}"
        )
        meta_path = qdir / "metadata.json"
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        mix_orig = mixture_original.get(sample_id, {}).get("mixture")
        span_mix = span_eval.get(sample_id, {}).get("mixture")
        pred_ev = span_eval.get(sample_id, {}).get("predicted_evidence")
        qwen_mix_orig = qwen_mixture_original.get(sample_id, {}).get("mixture")
        qwen_span_mix = qwen_span_eval.get(sample_id, {}).get("mixture")
        qwen_pred_ev = qwen_span_eval.get(sample_id, {}).get("predicted_evidence")
        phi_mix_orig = phi_mixture_original.get(sample_id, {}).get("mixture")
        phi_span_mix = phi_span_eval.get(sample_id, {}).get("mixture")
        phi_pred_ev = phi_span_eval.get(sample_id, {}).get("predicted_evidence")

        cases.append(
            {
                "id": sample_id,
                "scene_id": scene_id,
                "variant_id": record.get("variant_id"),
                "question_index": int(record["question_index"]),
                "question_type": str(record["question_type"]),
                "relation": str(record["relation"]),
                "no_evidence": bool(record["no_evidence"]),
                "answer": record.get("answer"),
                "question": record.get("question"),
                "span_question": pointer_manifest.get(sample_id, {}).get("question"),
                "answer_options": record.get("answer_options", []),
                "record": record,
                "pointer": pointer_item,
                "metadata": meta,
                "mixture_path": safe_path(dataset, str(record["mixture_path"])),
                "predicted_evidence_path": qdir / "predicted_evidence.wav",
                "predicted_residual_path": qdir / "predicted_residual.wav",
                "mixture_original": mix_orig,
                "mixture_span_text": span_mix,
                "predicted_evidence_eval": pred_ev,
                "qwen_mixture_original": qwen_mix_orig,
                "qwen_mixture_span_text": qwen_span_mix,
                "qwen_predicted_evidence_eval": qwen_pred_ev,
                "phi_mixture_original": phi_mix_orig,
                "phi_mixture_span_text": phi_span_mix,
                "phi_predicted_evidence_eval": phi_pred_ev,
            }
        )
    return {
        "cases": cases,
        "paths": {
            "original_manifest": str(original_path),
            "pointer_root": str(pointer),
            "af3_root": str(af3),
            "qwen_root": str(qwen),
            "phi_root": str(phi),
            "dataset_root": str(dataset),
        },
    }


def correctness_badge(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "not run"
    return "correct" if bool(row.get("correct")) else "wrong"


def prediction_text(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "—"
    return f"{label_text(row.get('predicted_answer'))} ({correctness_badge(row)})"


def case_filter_value(case: Mapping[str, Any]) -> str:
    mix = case.get("mixture_original")
    pred = case.get("predicted_evidence_eval")
    if mix and pred:
        mix_ok = bool(mix.get("correct"))
        pred_ok = bool(pred.get("correct"))
        if not mix_ok and pred_ok:
            return "Evidence fixes mixture"
        if mix_ok and not pred_ok:
            return "Evidence hurts mixture"
        if mix_ok and pred_ok:
            return "Both correct"
        return "Both wrong"
    if pred:
        return "Only evidence result"
    return "No AF3 result yet"


def pct_count(correct: int, total: int) -> str:
    if total <= 0:
        return "—"
    return f"{100.0 * correct / total:.1f}% ({correct}/{total})"


def pct_value(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def pp_delta(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):+.1f} pp"


@st.cache_data(show_spinner=False)
def load_dataset_dashboard(dataset_root: str, artifact_state: str) -> dict[str, Any]:
    _ = artifact_state
    root = Path(dataset_root)
    config = read_json(root / "dataset_config.json")
    validation = read_json(root / "validation_report.json")

    split_rows: list[dict[str, Any]] = []
    all_event_classes: set[str] = set()
    all_answer_classes: set[str] = set()
    source_datasets: set[str] = set()
    relation_counts_total: dict[str, int] = defaultdict(int)

    for split in SPLIT_ORDER:
        path = root / f"qces_{split}.jsonl"
        records = 0
        answerable = 0
        no_evidence = 0
        scenes: set[str] = set()
        families: set[str] = set()
        event_classes: set[str] = set()
        answer_classes: set[str] = set()
        relation_counts: dict[str, int] = defaultdict(int)

        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    records += 1
                    scenes.add(str(record.get("scene_id")))
                    families.add(str(record.get("scene_family_id")))
                    relation = str(record.get("relation") or "unknown")
                    relation_counts[relation] += 1
                    relation_counts_total[relation] += 1
                    if record.get("no_evidence"):
                        no_evidence += 1
                    else:
                        answerable += 1
                        answer = record.get("answer")
                        if answer:
                            answer_classes.add(str(answer))
                            all_answer_classes.add(str(answer))
                    for event in record.get("events", []):
                        label = event.get("label")
                        if label:
                            event_classes.add(str(label))
                            all_event_classes.add(str(label))
                        source_dataset = event.get("source_dataset")
                        if source_dataset:
                            source_datasets.add(str(source_dataset))

        split_rows.append(
            {
                "split": SPLIT_LABELS.get(split, split),
                "records": records,
                "scenes": len(scenes),
                "families": len(families),
                "event classes": len(event_classes),
                "answer/query classes": len(answer_classes),
                "answerable": answerable,
                "no-evidence": no_evidence,
                "no-evidence ratio ↓": pct_value(no_evidence / records if records else None),
                "first": relation_counts.get("first", 0),
                "before": relation_counts.get("before", 0),
                "after": relation_counts.get("after", 0),
            }
        )

    counts = config.get("counts", {})
    composition = config.get("composition", {})
    source = config.get("source", {})
    question_design = config.get("question_design", {})
    seen_labels = list(composition.get("seen_labels", []))
    heldout_labels = list(composition.get("heldout_labels", []))
    nuisance_labels = list(composition.get("nuisance_labels", []))
    semantic_answer_classes = set(seen_labels) | set(heldout_labels)
    class_rows = [
        {
            "class group": "event classes in audio",
            "count": len(all_event_classes),
            "meaning": "mọi label xuất hiện trong mixture, gồm answer/query + nuisance",
        },
        {
            "class group": "answer/query classes",
            "count": len(all_answer_classes) or len(semantic_answer_classes),
            "meaning": "label có thể là đáp án hoặc query chính",
        },
        {
            "class group": "seen answer/query classes",
            "count": len(seen_labels),
            "meaning": "label dùng ở train/val/test-IID/test-comp-OOD",
        },
        {
            "class group": "held-out label-OOD answer/query classes",
            "count": len(heldout_labels),
            "meaning": "label chỉ xuất hiện ở test-label-OOD",
        },
        {
            "class group": "nuisance/background classes",
            "count": len(nuisance_labels),
            "meaning": "label gây nhiễu, không phải đáp án chính",
        },
    ]
    quality_rows = [
        {
            "check": "scene-family overlap across splits ↓",
            "value": validation.get("scene_family_split_overlap_count_down", "—"),
        },
        {
            "check": "source overlap across splits ↓",
            "value": validation.get("source_split_overlap_count_down", "—"),
        },
        {
            "check": "held-out label leakage into train ↓",
            "value": validation.get("heldout_label_train_presence_count_down", "—"),
        },
        {
            "check": "evidence + residual reconstruction error ↓",
            "value": (
                validation.get("maximum_audio_errors_down", {}).get(
                    "evidence_plus_residual", "—"
                )
                if validation
                else "—"
            ),
        },
        {
            "check": "semantic balance / paper eligibility",
            "value": config.get("semantic_balance", {}).get("paper_eligibility", "—"),
        },
        {
            "check": "audio redistribution ready",
            "value": validation.get("audio_redistribution_ready", "—"),
        },
    ]
    overview = {
        "records": int(counts.get("records", validation.get("records", 0)) or 0),
        "scenes": int(counts.get("scenes", 0) or 0),
        "families": int(
            counts.get("scene_families", validation.get("scene_families_validated", 0))
            or 0
        ),
        "questions_per_scene": int(counts.get("questions_per_scene", 0) or 0),
        "event_classes": len(all_event_classes),
        "answer_classes": len(all_answer_classes) or len(semantic_answer_classes),
        "source_dataset": source.get("dataset", "—"),
        "dataset_revision": source.get("dataset_revision", "—"),
        "relations": ", ".join(question_design.get("relations", [])) or "—",
        "fingerprint": str(validation.get("artifact_fingerprint_sha256", "—"))[:16],
        "all_event_classes": sorted(all_event_classes),
        "all_answer_classes": sorted(all_answer_classes),
        "seen_labels": sorted(seen_labels),
        "heldout_labels": sorted(heldout_labels),
        "nuisance_labels": sorted(nuisance_labels),
        "source_datasets": sorted(source_datasets),
        "relation_counts": dict(relation_counts_total),
    }
    return {
        "overview": overview,
        "split_df": pd.DataFrame(split_rows),
        "class_df": pd.DataFrame(class_rows),
        "quality_df": pd.DataFrame(quality_rows),
    }


@st.cache_data(show_spinner=False)
def load_rankcal_benchmark(report_path: str, artifact_state: str) -> pd.DataFrame:
    _ = artifact_state
    report = read_json(Path(report_path))
    eval_results = report.get("eval_results", {})
    policy = "max_answer_min_noev_0.57"
    rows = []
    for split in ["val", "test_iid", "test_compositional_ood", "test_label_ood"]:
        policies = eval_results.get(split, {})
        metrics = policies.get(policy) or policies.get("max_answer_min_noev_0.57") or {}
        if not metrics:
            continue
        record_count = int(metrics.get("record_count", 0) or 0)
        answerable_count = int(metrics.get("answerable_count", 0) or 0)
        noev_count = int(metrics.get("no_evidence_count", 0) or 0)
        answer_acc = float(metrics.get("answer_accuracy_↑", 0.0) or 0.0)
        noev_pure = float(metrics.get("no_evidence_pure_accuracy_↑", 0.0) or 0.0)
        all_acc = (
            (answer_acc * answerable_count + noev_pure * noev_count) / record_count
            if record_count
            else None
        )
        rows.append(
            {
                "split": SPLIT_LABELS.get(split, split),
                "policy": policy,
                "N": record_count,
                "All exact ↑": pct_value(all_acc),
                "Answerable answer ↑": pct_value(answer_acc),
                "No-evidence pure ↑": pct_value(noev_pure),
                "No-evidence decision ↑": pct_value(
                    metrics.get("no_evidence_decision_accuracy_↑")
                ),
                "Abstain on answerable ↓": pct_value(
                    metrics.get("abstain_rate_on_answerable_↓")
                ),
                "Answer-span IoU ↑": f"{float(metrics.get('span_iou_mean_↑', 0.0)):.3f}",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_rendered_evidence_reports(
    fast_report_path: str,
    val_report_path: str,
    artifact_state: str,
) -> pd.DataFrame:
    _ = artifact_state
    rows = []
    for label, path in [
        ("fast384 evidence subset", Path(fast_report_path)),
        ("val-full evidence", Path(val_report_path)),
    ]:
        report = read_json(path)
        summaries = report.get("summaries", {})
        metrics = summaries.get("predicted_context_span_text", {})
        if not metrics:
            continue
        rows.append(
            {
                "evidence artifact": label,
                "N": int(metrics.get("rendered_records_↑", 0) or 0),
                "Ours answerable ↑": pct_value(
                    metrics.get("rankcal_answer_accuracy_on_rendered_answerable_↑")
                ),
                "Ours no-evidence ↑": pct_value(
                    metrics.get("rankcal_no_evidence_accuracy_on_rendered_noev_↑")
                ),
                "note": "metrics trực tiếp của evidence/pointer trước khi gọi AF3/Qwen/Phi",
            }
        )
    return pd.DataFrame(rows)


def interval_iou(
    predicted: Sequence[Mapping[str, Any]] | None,
    gold: Sequence[Sequence[float]] | None,
) -> float | None:
    def normalize_dict_spans(spans: Sequence[Mapping[str, Any]] | None) -> list[tuple[float, float]]:
        out = []
        for span in spans or []:
            start = float(span["onset_seconds"])
            end = float(span["offset_seconds"])
            if end > start:
                out.append((start, end))
        return merge_intervals(out)

    def normalize_pair_spans(spans: Sequence[Sequence[float]] | None) -> list[tuple[float, float]]:
        out = []
        for start, end in spans or []:
            start = float(start)
            end = float(end)
            if end > start:
                out.append((start, end))
        return merge_intervals(out)

    def total_length(intervals: Sequence[tuple[float, float]]) -> float:
        return sum(end - start for start, end in intervals)

    pred_intervals = normalize_dict_spans(predicted)
    gold_intervals = normalize_pair_spans(gold)
    if not pred_intervals and not gold_intervals:
        return None

    intersection = 0.0
    pred_index = 0
    gold_index = 0
    while pred_index < len(pred_intervals) and gold_index < len(gold_intervals):
        pred_start, pred_end = pred_intervals[pred_index]
        gold_start, gold_end = gold_intervals[gold_index]
        start = max(pred_start, gold_start)
        end = min(pred_end, gold_end)
        if end > start:
            intersection += end - start
        if pred_end < gold_end:
            pred_index += 1
        else:
            gold_index += 1

    union = total_length(pred_intervals) + total_length(gold_intervals) - intersection
    if union <= 0.0:
        return None
    return intersection / union


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def event_intervals_for_ids(
    record: Mapping[str, Any],
    event_ids: Sequence[str] | None,
) -> list[tuple[float, float]]:
    wanted = {str(event_id) for event_id in event_ids or []}
    intervals = []
    for event in record.get("events", []):
        if str(event.get("event_id")) not in wanted:
            continue
        start = float(event.get("onset_seconds", 0.0))
        end = float(event.get("offset_seconds", 0.0))
        if end > start:
            intervals.append((start, end))
    return merge_intervals(intervals)


def gold_evidence_intervals(record: Mapping[str, Any]) -> list[tuple[float, float]]:
    intervals = event_intervals_for_ids(record, record.get("evidence_event_ids"))
    if intervals:
        return intervals
    fallback = []
    for start, end in list(record.get("anchor_intervals", [])) + list(
        record.get("answer_intervals", [])
    ):
        start = float(start)
        end = float(end)
        if end > start:
            fallback.append((start, end))
    return merge_intervals(fallback)


def predicted_context_spans(case: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    metadata = case.get("metadata") or {}
    if metadata.get("selected_events"):
        return metadata.get("selected_events") or []
    pointer = case.get("pointer") or {}
    return pointer.get("inferred_answer_spans") or []


def fmt_iou(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def internal_correct(case: Mapping[str, Any]) -> bool | None:
    pointer = case.get("pointer") or {}
    if not pointer:
        return None
    if case.get("no_evidence"):
        return bool(pointer.get("rankcal_predicted_no_evidence"))
    return (not bool(pointer.get("rankcal_predicted_no_evidence"))) and bool(
        pointer.get("rankcal_answer_correct")
    )


def internal_prediction(case: Mapping[str, Any]) -> str:
    pointer = case.get("pointer") or {}
    if not pointer:
        return "not available"
    if pointer.get("rankcal_predicted_no_evidence"):
        return "no_evidence"
    return label_text(pointer.get("rankcal_predicted_answer"))


def internal_eval_tables(cases: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = [case for case in cases if case.get("pointer")]
    answerable = [case for case in valid if not case.get("no_evidence")]
    no_evidence = [case for case in valid if case.get("no_evidence")]

    def count_correct(rows: Sequence[Mapping[str, Any]]) -> int:
        return sum(bool(internal_correct(case)) for case in rows)

    false_noev_answerable = sum(
        bool((case.get("pointer") or {}).get("rankcal_predicted_no_evidence"))
        for case in answerable
    )
    false_answer_noev = len(no_evidence) - count_correct(no_evidence)
    answer_ious = [
        value
        for value in (
            interval_iou(
                (case.get("pointer") or {}).get("inferred_answer_spans"),
                (case.get("record") or {}).get("answer_intervals"),
            )
            for case in answerable
        )
        if value is not None
    ]
    evidence_ious = [
        value
        for value in (
            interval_iou(
                predicted_context_spans(case),
                gold_evidence_intervals(case.get("record") or {}),
            )
            for case in answerable
        )
        if value is not None
    ]
    summary = pd.DataFrame(
        [
            {
                "evaluation": "Our RankCal/pointer, không qua AF3",
                "N": len(valid),
                "All ↑": pct_count(count_correct(valid), len(valid)),
                "Answerable ↑": pct_count(count_correct(answerable), len(answerable)),
                "No-evidence ↑": pct_count(count_correct(no_evidence), len(no_evidence)),
                "False no-evidence on answerable ↓": pct_count(
                    false_noev_answerable, len(answerable)
                ),
                "False answer on no-evidence ↓": pct_count(
                    false_answer_noev, len(no_evidence)
                ),
                "Answer-span IoU ↑": (
                    f"{sum(answer_ious) / len(answer_ious):.3f}" if answer_ious else "—"
                ),
                "Evidence-context IoU ↑": (
                    f"{sum(evidence_ious) / len(evidence_ious):.3f}"
                    if evidence_ious
                    else "—"
                ),
            }
        ]
    )

    relation_rows = []
    for relation in sorted({str(case.get("relation")) for case in valid}):
        rows = [case for case in valid if str(case.get("relation")) == relation]
        rel_answerable = [case for case in rows if not case.get("no_evidence")]
        rel_no_evidence = [case for case in rows if case.get("no_evidence")]
        rel_answer_ious = [
            value
            for value in (
                interval_iou(
                    (case.get("pointer") or {}).get("inferred_answer_spans"),
                    (case.get("record") or {}).get("answer_intervals"),
                )
                for case in rel_answerable
            )
            if value is not None
        ]
        rel_evidence_ious = [
            value
            for value in (
                interval_iou(
                    predicted_context_spans(case),
                    gold_evidence_intervals(case.get("record") or {}),
                )
                for case in rel_answerable
            )
            if value is not None
        ]
        relation_rows.append(
            {
                "relation": relation,
                "N": len(rows),
                "All ↑": pct_count(count_correct(rows), len(rows)),
                "Answerable ↑": pct_count(
                    count_correct(rel_answerable), len(rel_answerable)
                ),
                "No-evidence ↑": pct_count(
                    count_correct(rel_no_evidence), len(rel_no_evidence)
                ),
                "Answer-span IoU ↑": (
                    f"{sum(rel_answer_ious) / len(rel_answer_ious):.3f}"
                    if rel_answer_ious
                    else "—"
                ),
                "Evidence-context IoU ↑": (
                    f"{sum(rel_evidence_ious) / len(rel_evidence_ious):.3f}"
                    if rel_evidence_ious
                    else "—"
                ),
            }
        )

    return summary, pd.DataFrame(relation_rows)


def audioqa_aggregate_tables(cases: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_specs = [
        (
            "mixture + original question",
            "mixture_original",
            "qwen_mixture_original",
            "phi_mixture_original",
            "mixture gốc",
            "câu hỏi gốc",
            "AF3 nghe toàn bộ audio gốc và trả lời câu hỏi gốc.",
        ),
        (
            "mixture + span-text prompt (from ours)",
            "mixture_span_text",
            "qwen_mixture_span_text",
            "phi_mixture_span_text",
            "mixture gốc",
            "prompt có predicted answer span",
            "AF3 vẫn nghe audio gốc, nhưng text được viết lại để chỉ rõ khoảng thời gian hệ mình dự đoán là đáp án.",
        ),
        (
            "predicted evidence (from ours) + span-text prompt (from ours)",
            "predicted_evidence_eval",
            "qwen_predicted_evidence_eval",
            "phi_predicted_evidence_eval",
            "predicted evidence của mình",
            "prompt có predicted answer span",
            "AF3 chỉ nghe phần evidence hệ mình tách ra, kèm prompt chỉ rõ khoảng thời gian đáp án.",
        ),
    ]

    summary_rows = []
    for evaluator, key_index, note_prefix in [
        ("AF3", 1, "AF3"),
        ("Qwen2-Audio", 2, "Qwen2-Audio"),
        ("Phi-4mm", 3, "Phi-4mm"),
    ]:
        for (
            label,
            af3_key,
            qwen_key,
            phi_key,
            audio_desc,
            prompt_desc,
            note,
        ) in condition_specs:
            key = {1: af3_key, 2: qwen_key, 3: phi_key}[key_index]
            rows = [case.get(key) for case in cases if case.get(key)]
            answerable = [row for row in rows if not row.get("no_evidence")]
            no_evidence = [row for row in rows if row.get("no_evidence")]
            correct = sum(bool(row.get("correct")) for row in rows)
            answerable_correct = sum(bool(row.get("correct")) for row in answerable)
            no_evidence_correct = sum(bool(row.get("correct")) for row in no_evidence)
            false_noev_on_answerable = sum(
                str(row.get("predicted_answer")) == "no_evidence" for row in answerable
            )
            false_answer_on_noev = len(no_evidence) - no_evidence_correct
            mean_gold_probability = (
                sum(float(row.get("gold_option_probability", 0.0)) for row in rows) / len(rows)
                if rows
                else None
            )
            summary_rows.append(
                {
                    "evaluator": evaluator,
                    "input": label,
                    "chú thích": note.replace("AF3", note_prefix, 1),
                    "audio": audio_desc,
                    "text/prompt": prompt_desc,
                    "N": len(rows),
                    "All ↑": pct_count(correct, len(rows)),
                    "Answerable ↑": pct_count(answerable_correct, len(answerable)),
                    "No-evidence ↑": pct_count(no_evidence_correct, len(no_evidence)),
                    "False no-evidence on answerable ↓": pct_count(
                        false_noev_on_answerable, len(answerable)
                    ),
                    "False answer on no-evidence ↓": pct_count(
                        false_answer_on_noev, len(no_evidence)
                    ),
                    "Mean P(gold) ↑": (
                        f"{mean_gold_probability:.3f}"
                        if mean_gold_probability is not None
                        else "—"
                    ),
                    "Answer-span IoU ↑": "—",
                    "Evidence-context IoU ↑": "—",
                }
            )

    valid = [case for case in cases if case.get("pointer")]
    answerable = [case for case in valid if not case.get("no_evidence")]
    no_evidence = [case for case in valid if case.get("no_evidence")]
    internal_correct_all = sum(bool(internal_correct(case)) for case in valid)
    internal_correct_answerable = sum(bool(internal_correct(case)) for case in answerable)
    internal_correct_noev = sum(bool(internal_correct(case)) for case in no_evidence)
    internal_false_noev_on_answerable = sum(
        bool((case.get("pointer") or {}).get("rankcal_predicted_no_evidence"))
        for case in answerable
    )
    internal_false_answer_on_noev = len(no_evidence) - internal_correct_noev
    answer_ious = [
        value
        for value in (
            interval_iou(
                (case.get("pointer") or {}).get("inferred_answer_spans"),
                (case.get("record") or {}).get("answer_intervals"),
            )
            for case in answerable
        )
        if value is not None
    ]
    evidence_ious = [
        value
        for value in (
            interval_iou(
                predicted_context_spans(case),
                gold_evidence_intervals(case.get("record") or {}),
            )
            for case in answerable
        )
        if value is not None
    ]
    summary_rows.append(
        {
            "evaluator": "ours",
            "input": "ours internal RankCal/pointer",
            "chú thích": "Kết quả gốc của hệ mình: parser/planner chọn đáp án và span trực tiếp, không qua AF3.",
            "audio": "không gọi AF3; dùng event/evidence proposal nội bộ",
            "text/prompt": "question parser + RankCal planner",
            "N": len(valid),
            "All ↑": pct_count(internal_correct_all, len(valid)),
            "Answerable ↑": pct_count(internal_correct_answerable, len(answerable)),
            "No-evidence ↑": pct_count(internal_correct_noev, len(no_evidence)),
            "False no-evidence on answerable ↓": pct_count(
                internal_false_noev_on_answerable, len(answerable)
            ),
            "False answer on no-evidence ↓": pct_count(
                internal_false_answer_on_noev, len(no_evidence)
            ),
            "Mean P(gold) ↑": "—",
            "Answer-span IoU ↑": (
                f"{sum(answer_ious) / len(answer_ious):.3f}" if answer_ious else "—"
            ),
            "Evidence-context IoU ↑": (
                f"{sum(evidence_ious) / len(evidence_ious):.3f}" if evidence_ious else "—"
            ),
        }
    )

    pair_specs = [
        (
            "mixture + original question → predicted evidence (from ours) + span-text prompt (from ours)",
            "mixture_original",
            "predicted_evidence_eval",
            "qwen_mixture_original",
            "qwen_predicted_evidence_eval",
            "phi_mixture_original",
            "phi_predicted_evidence_eval",
        ),
        (
            "mixture + span-text prompt (from ours) → predicted evidence (from ours) + span-text prompt (from ours)",
            "mixture_span_text",
            "predicted_evidence_eval",
            "qwen_mixture_span_text",
            "qwen_predicted_evidence_eval",
            "phi_mixture_span_text",
            "phi_predicted_evidence_eval",
        ),
        (
            "mixture + original question → mixture + span-text prompt (from ours)",
            "mixture_original",
            "mixture_span_text",
            "qwen_mixture_original",
            "qwen_mixture_span_text",
            "phi_mixture_original",
            "phi_mixture_span_text",
        ),
    ]
    pair_rows = []
    for evaluator, base_index, new_index in [
        ("AF3", 1, 2),
        ("Qwen2-Audio", 3, 4),
        ("Phi-4mm", 5, 6),
    ]:
        for spec in pair_specs:
            label = spec[0]
            base_key = spec[base_index]
            new_key = spec[new_index]
            pairs = [
                (case.get(base_key), case.get(new_key))
                for case in cases
                if case.get(base_key) and case.get(new_key)
            ]
            base_correct = sum(bool(base.get("correct")) for base, _ in pairs)
            new_correct = sum(bool(new.get("correct")) for _, new in pairs)
            wrong_to_correct = sum(
                (not bool(base.get("correct"))) and bool(new.get("correct"))
                for base, new in pairs
            )
            correct_to_wrong = sum(
                bool(base.get("correct")) and (not bool(new.get("correct")))
                for base, new in pairs
            )
            same_correct = sum(
                bool(base.get("correct")) and bool(new.get("correct"))
                for base, new in pairs
            )
            same_wrong = sum(
                (not bool(base.get("correct"))) and (not bool(new.get("correct")))
                for base, new in pairs
            )
            if pairs:
                delta = 100.0 * (new_correct - base_correct) / len(pairs)
                gold_prob_delta = sum(
                    float(new.get("gold_option_probability", 0.0))
                    - float(base.get("gold_option_probability", 0.0))
                    for base, new in pairs
                ) / len(pairs)
            else:
                delta = 0.0
                gold_prob_delta = 0.0
            pair_rows.append(
                {
                    "evaluator": evaluator,
                    "comparison": label,
                    "N": len(pairs),
                    "Base acc ↑": pct_count(base_correct, len(pairs)),
                    "New acc ↑": pct_count(new_correct, len(pairs)),
                    "Δ all ↑": f"{delta:+.1f} pp" if pairs else "—",
                    "wrong→correct ↑": wrong_to_correct,
                    "correct→wrong ↓": correct_to_wrong,
                    "same correct": same_correct,
                    "same wrong": same_wrong,
                    "Δ Mean P(gold) ↑": f"{gold_prob_delta:+.3f}" if pairs else "—",
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def event_table(record: Mapping[str, Any]) -> pd.DataFrame:
    events = []
    for event in record.get("events", []):
        events.append(
            {
                "event_id": event.get("event_id"),
                "label": label_text(event.get("label")),
                "occ": event.get("occurrence_index"),
                "start": float(event.get("onset_seconds", 0.0)),
                "end": float(event.get("offset_seconds", 0.0)),
                "duration": float(event.get("offset_seconds", 0.0))
                - float(event.get("onset_seconds", 0.0)),
            }
        )
    return pd.DataFrame(events)


def compact_scene_events(record: Mapping[str, Any]) -> str:
    rows = []
    for event in sorted(
        record.get("events", []),
        key=lambda item: float(item.get("onset_seconds", 0.0)),
    ):
        label = label_text(event.get("label"))
        occurrence = event.get("occurrence_index")
        start = float(event.get("onset_seconds", 0.0))
        end = float(event.get("offset_seconds", 0.0))
        rows.append(f"{label} #{occurrence} ({start:.2f}–{end:.2f}s)")
    return " → ".join(rows) if rows else "—"


def option_summary(options: Sequence[Any], gold_answer: Any) -> pd.DataFrame:
    rows = []
    for index, option in enumerate(options):
        option_text = label_text(str(option))
        rows.append(
            {
                "option": chr(ord("A") + index),
                "label": option_text,
                "gold": str(option) == str(gold_answer),
            }
        )
    return pd.DataFrame(rows)


def correctness_text(value: bool | None) -> str:
    if value is None:
        return "not available"
    return "correct" if value else "wrong"


def spans_table(spans: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, span in enumerate(spans, start=1):
        rows.append(
            {
                "#": index,
                "label": label_text(span.get("label")),
                "span": fmt_span(
                    float(span["onset_seconds"]), float(span["offset_seconds"])
                ),
                "confidence": span.get("confidence"),
            }
        )
    return pd.DataFrame(rows)


def answer_interval_table(record: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name, ids_key, intervals_key in [
        ("gold anchor", "anchor_event_ids", "anchor_intervals"),
        ("gold answer", "answer_event_ids", "answer_intervals"),
        ("gold target evidence", "evidence_event_ids", None),
    ]:
        ids = record.get(ids_key, [])
        intervals = (
            gold_evidence_intervals(record)
            if intervals_key is None
            else record.get(intervals_key, [])
        )
        if not ids and not intervals:
            rows.append({"role": name, "event_ids": "—", "span": "—"})
            continue
        span_text = ", ".join(fmt_span(float(a), float(b)) for a, b in intervals)
        rows.append({"role": name, "event_ids": ", ".join(ids), "span": span_text})
    return pd.DataFrame(rows)


def render_audio(label: str, path: Path, caption: str | None = None) -> None:
    st.caption(label)
    if path.is_file():
        st.audio(str(path))
        if caption:
            st.caption(caption)
    else:
        st.warning(f"Missing audio: {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    st.set_page_config(
        page_title="QCES Pointer Evidence Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("QCES predicted context-span evidence demo")
    st.caption(
        "Một trang để nghe và kiểm tra từng case: mixture, predicted evidence, "
        "predicted residual, spans, options và AF3 outputs nếu đã chạy."
    )

    if st.sidebar.button("Reload artifacts / clear cache"):
        st.cache_data.clear()

    artifact_state = artifact_fingerprint(
        args.original_manifest,
        args.pointer_root,
        args.af3_root,
        args.qwen_root,
        args.phi_root,
    )
    bundle = load_bundle(
        str(args.original_manifest),
        str(args.pointer_root),
        str(args.af3_root),
        str(args.qwen_root),
        str(args.phi_root),
        str(args.dataset_root),
        artifact_state,
    )
    cases = bundle["cases"]
    if not cases:
        st.error("Không tìm thấy case nào. Kiểm tra artifact paths ở sidebar.")
        st.stop()

    dataset_state = dataset_fingerprint(args.dataset_root)
    dataset_dashboard = load_dataset_dashboard(
        str(args.dataset_root),
        dataset_state,
    )
    rankcal_state = files_fingerprint([args.rankcal_report])
    benchmark_df = load_rankcal_benchmark(str(args.rankcal_report), rankcal_state)
    fast_evidence_report = args.pointer_root / "predictions/evaluation_report.json"
    evidence_state = files_fingerprint([fast_evidence_report, args.val_evidence_report])
    rendered_evidence_df = load_rendered_evidence_reports(
        str(fast_evidence_report),
        str(args.val_evidence_report),
        evidence_state,
    )

    with st.sidebar:
        st.header("Filters")
        st.caption("Artifacts")
        st.code(
            "\n".join(
                [
                    f"original_manifest: {bundle['paths']['original_manifest']}",
                    f"pointer_root: {bundle['paths']['pointer_root']}",
                    f"af3_root: {bundle['paths']['af3_root']}",
                    f"qwen_root: {bundle['paths']['qwen_root']}",
                    f"phi_root: {bundle['paths']['phi_root']}",
                    f"dataset_root: {bundle['paths']['dataset_root']}",
                    f"rankcal_report: {args.rankcal_report}",
                    f"val_evidence_report: {args.val_evidence_report}",
                ]
            ),
            language="text",
        )
        relations = ["all"] + sorted({case["relation"] for case in cases})
        relation = st.selectbox("Relation", relations, index=0)
        answerability = st.selectbox(
            "Answerability", ["all", "answerable", "no-evidence"], index=0
        )
        outcome_options = ["all"] + sorted({case_filter_value(case) for case in cases})
        outcome = st.selectbox("AF3 outcome", outcome_options, index=0)
        search = st.text_input("Search id / label / question", "")
        show_gold = st.checkbox("Show gold answer/spans", value=True)

    _, internal_relation_df = internal_eval_tables(cases)
    summary_df, pair_df = audioqa_aggregate_tables(cases)

    st.subheader("Dataset / class inventory")
    overview = dataset_dashboard["overview"]
    dcols = st.columns(6)
    dcols[0].metric("Records", f"{overview['records']:,}")
    dcols[1].metric("Scenes", f"{overview['scenes']:,}")
    dcols[2].metric("Families", f"{overview['families']:,}")
    dcols[3].metric("Event classes", f"{overview['event_classes']}")
    dcols[4].metric("Answer/query classes", f"{overview['answer_classes']}")
    dcols[5].metric("Questions/scene", f"{overview['questions_per_scene']}")
    st.caption(
        f"Source: {overview['source_dataset']} · revision: {overview['dataset_revision']} · "
        f"relations: {overview['relations']} · artifact fingerprint: {overview['fingerprint']}…"
    )
    st.dataframe(dataset_dashboard["class_df"], use_container_width=True, hide_index=True)
    with st.expander("Dataset splits / answerability / relation counts"):
        st.dataframe(dataset_dashboard["split_df"], use_container_width=True, hide_index=True)
    with st.expander("Dataset validation checks"):
        st.dataframe(dataset_dashboard["quality_df"], use_container_width=True, hide_index=True)
    with st.expander("Class labels currently used"):
        label_cols = st.columns(3)
        with label_cols[0]:
            st.markdown("Seen answer/query labels")
            st.write(", ".join(label_text(x) for x in overview["seen_labels"]) or "—")
        with label_cols[1]:
            st.markdown("Held-out label-OOD labels")
            st.write(", ".join(label_text(x) for x in overview["heldout_labels"]) or "—")
        with label_cols[2]:
            st.markdown("Nuisance/background labels")
            st.write(", ".join(label_text(x) for x in overview["nuisance_labels"]) or "—")

    st.subheader("Full benchmark metrics for ours")
    st.caption(
        "Đây là kết quả hệ RankCal/pointer của mình trên các split full. "
        "`All exact ↑` tính đúng cả answerable và no-evidence. "
        "`Answer-span IoU ↑` đo overlap với span đáp án; `Abstain ↓` thấp hơn là tốt."
    )
    if benchmark_df.empty:
        st.warning("Không đọc được rankcal benchmark report.")
    else:
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

    st.subheader("Rendered evidence metrics for ours")
    st.caption(
        "`fast384 evidence subset` là 384 case đang dùng cho demo/nghe và external AudioQA. "
        "`val-full evidence` là toàn bộ validation split đã render predicted evidence."
    )
    if rendered_evidence_df.empty:
        st.warning("Không đọc được rendered evidence report.")
    else:
        st.dataframe(rendered_evidence_df, use_container_width=True, hide_index=True)

    st.subheader("External AudioQA metrics on fast384")
    st.caption(
        "Cột `input` ghi theo dạng audio + text/prompt. Các dòng AF3/Qwen/Phi đọc audio+prompt. "
        "Dòng `ours internal RankCal/pointer` là "
        "đánh giá trực tiếp output của hệ mình so với annotation, không gọi AF3. "
        "`N` là số câu đã chạy; `All ↑` là số đúng / tổng số câu."
    )
    compact_summary_df = summary_df[
        ["evaluator", "input", "chú thích", "N", "All ↑"]
    ]
    st.dataframe(compact_summary_df, use_container_width=True, hide_index=True)
    with st.expander("Detailed aggregate metrics"):
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    with st.expander("Internal ours by relation"):
        st.dataframe(internal_relation_df, use_container_width=True, hide_index=True)
    with st.expander("AudioQA pairwise change"):
        st.dataframe(pair_df, use_container_width=True, hide_index=True)

    st.caption(
        "Sanity check hiện tại: `mixture + span-text prompt (from ours)` và "
        "`predicted evidence (from ours) + span-text prompt (from ours)` đều đã có N=384. "
        "Nếu thấy 201/384 hoặc 203/384 thì đó là số câu đúng, không phải số câu đã chạy."
    )

    filtered = []
    search_norm = search.strip().lower()
    for case in cases:
        if relation != "all" and case["relation"] != relation:
            continue
        if answerability == "answerable" and case["no_evidence"]:
            continue
        if answerability == "no-evidence" and not case["no_evidence"]:
            continue
        if outcome != "all" and case_filter_value(case) != outcome:
            continue
        if search_norm:
            haystack = " ".join(
                [
                    str(case["id"]),
                    str(case["question"]),
                    str(case.get("span_question")),
                    str(case.get("answer")),
                    " ".join(map(str, case.get("answer_options", []))),
                    " ".join(case.get("pointer", {}).get("selected_labels", [])),
                ]
            ).lower()
            if search_norm not in haystack:
                continue
        filtered.append(case)

    st.subheader("Case selection")
    st.write(f"Showing {len(filtered)} / {len(cases)} cases.")
    if not filtered:
        st.stop()

    def case_label(case: Mapping[str, Any]) -> str:
        pred = case.get("predicted_evidence_eval")
        mix = case.get("mixture_original")
        return (
            f"{case['id']} · {case['relation']} · "
            f"{'noev' if case['no_evidence'] else label_text(case['answer'])} · "
            f"{case_filter_value(case)} · "
            f"M:{correctness_badge(mix)} E:{correctness_badge(pred)}"
        )

    selected_label = st.selectbox("Case", [case_label(case) for case in filtered])
    case = filtered[[case_label(item) for item in filtered].index(selected_label)]

    record = case["record"]
    pointer = case.get("pointer", {})
    meta = case.get("metadata", {})
    inferred_spans = pointer.get("inferred_answer_spans") or meta.get(
        "inferred_predicted_answer_spans", []
    )
    selected_events = meta.get("selected_events") or [
        {"label": label}
        for label in pointer.get("selected_labels", [])
    ]
    case_answer_iou = interval_iou(
        pointer.get("inferred_answer_spans"),
        record.get("answer_intervals"),
    )
    case_evidence_iou = interval_iou(
        predicted_context_spans(case),
        gold_evidence_intervals(record),
    )

    st.subheader("Case overview")
    st.markdown("Audio gồm những sự kiện gì")
    st.info(compact_scene_events(record))
    overview_left, overview_right = st.columns([1.15, 1])
    with overview_left:
        st.markdown("Câu hỏi")
        st.info(str(record.get("question")))
        st.markdown("Answer options")
        st.dataframe(
            option_summary(case.get("answer_options", []), case.get("answer")),
            use_container_width=True,
            hide_index=True,
        )
    with overview_right:
        gold_text = "no_evidence" if case["no_evidence"] else label_text(case["answer"])
        ours_text = internal_prediction(case)
        ours_correct = internal_correct(case)
        st.markdown("Kết quả chính")
        result_df = pd.DataFrame(
            [
                {
                    "item": "Gold answer",
                    "value": gold_text,
                },
                {
                    "item": "Ours prediction",
                    "value": ours_text,
                },
                {
                    "item": "Ours correct?",
                    "value": correctness_text(ours_correct),
                },
                {
                    "item": "Relation",
                    "value": case["relation"],
                },
                {
                    "item": "Predicted no-evidence?",
                    "value": bool(pointer.get("rankcal_predicted_no_evidence")),
                },
                {
                    "item": "Answer-span IoU",
                    "value": fmt_iou(case_answer_iou),
                },
                {
                    "item": "Target-evidence/context IoU",
                    "value": fmt_iou(case_evidence_iou),
                },
            ]
        )
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        if ours_correct is True:
            st.success("Ours dự đoán đúng case này.")
        elif ours_correct is False:
            st.error("Ours dự đoán sai case này.")
        else:
            st.warning("Không có prediction nội bộ cho case này.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Gold", "no_evidence" if case["no_evidence"] else label_text(case["answer"]))
    c2.metric(
        "RankCal answer",
        label_text(pointer.get("rankcal_predicted_answer")),
        "no_evidence" if pointer.get("rankcal_predicted_no_evidence") else "",
    )
    c3.metric("Mixture AF3", prediction_text(case.get("mixture_original")))
    c4.metric("Evidence AF3", prediction_text(case.get("predicted_evidence_eval")))
    c5.metric("Evidence Qwen", prediction_text(case.get("qwen_predicted_evidence_eval")))
    c6.metric("Evidence Phi", prediction_text(case.get("phi_predicted_evidence_eval")))

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Prompt details")
        st.markdown("Original question")
        st.info(str(record.get("question")))
        st.markdown("Span-text prompt used by predicted evidence")
        st.info(str(case.get("span_question") or "not available yet"))
        st.markdown("Answer options")
        options_df = pd.DataFrame(
            {
                "option_index": list(range(len(case.get("answer_options", [])))),
                "option": [label_text(x) for x in case.get("answer_options", [])],
                "is_gold": [
                    str(x) == str(case.get("answer"))
                    for x in case.get("answer_options", [])
                ],
            }
        )
        st.dataframe(options_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Predicted planner evidence")
        st.dataframe(
            spans_table(selected_events),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("Inferred predicted answer span")
        st.dataframe(
            spans_table(inferred_spans),
            use_container_width=True,
            hide_index=True,
        )
        if show_gold:
            st.markdown("Gold spans for debugging")
            st.dataframe(
                answer_interval_table(record),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("Audio")
    a1, a2, a3 = st.columns(3)
    with a1:
        render_audio("Mixture · audio gốc", Path(case["mixture_path"]))
    with a2:
        render_audio(
            "Predicted evidence · context + predicted answer span",
            Path(case["predicted_evidence_path"]),
        )
    with a3:
        render_audio("Predicted residual", Path(case["predicted_residual_path"]))

    st.divider()
    st.subheader("AudioQA outputs + ours")
    st.info(
        "Các dòng AF3/Qwen là downstream Audio-QA. Dòng cuối là dự đoán nội bộ "
        "của hệ mình, lấy trực tiếp từ RankCal/pointer nên không gọi external AudioQA. "
        "`question_only` không truyền audio và đã bị bỏ khỏi demo/evaluation chính."
    )
    input_descriptions = pd.DataFrame(
        [
            {
                "input": "mixture + original question",
                "audio": "mixture gốc",
                "text/prompt": "câu hỏi gốc",
                "mục đích": "baseline Audio-QA trực tiếp trên audio gốc",
            },
            {
                "input": "mixture + span-text prompt (from ours)",
                "audio": "mixture gốc",
                "text/prompt": "prompt có predicted answer span",
                "mục đích": "kiểm tra chỉ thêm chỉ dẫn thời gian có giúp không",
            },
            {
                "input": "predicted evidence (from ours) + span-text prompt (from ours)",
                "audio": "predicted evidence của mình",
                "text/prompt": "prompt có predicted answer span",
                "mục đích": "kiểm tra evidence tách ra có giúp trả lời không",
            },
            {
                "input": "ours internal RankCal/pointer",
                "audio": "không đưa vào external AudioQA",
                "text/prompt": "question parser + RankCal planner",
                "mục đích": "đáp án hệ mình tự chọn trước khi đưa evidence qua AF3/Qwen",
            },
        ]
    )
    st.dataframe(input_descriptions, use_container_width=True, hide_index=True)
    rows = []
    for evaluator, name, note, audio_desc, prompt_desc, row in [
        (
            "AF3",
            "mixture + original question",
            "AF3 nghe toàn bộ audio gốc và trả lời câu hỏi gốc.",
            "mixture gốc",
            "câu hỏi gốc",
            case.get("mixture_original"),
        ),
        (
            "AF3",
            "mixture + span-text prompt (from ours)",
            "AF3 vẫn nghe audio gốc, nhưng prompt chỉ rõ predicted answer span.",
            "mixture gốc",
            "prompt có predicted answer span",
            case.get("mixture_span_text"),
        ),
        (
            "AF3",
            "predicted evidence (from ours) + span-text prompt (from ours)",
            "AF3 chỉ nghe evidence hệ mình tách ra, kèm prompt chỉ rõ predicted answer span.",
            "predicted evidence của mình",
            "prompt có predicted answer span",
            case.get("predicted_evidence_eval"),
        ),
        (
            "Qwen2-Audio",
            "mixture + original question",
            "Qwen2-Audio nghe toàn bộ audio gốc và trả lời câu hỏi gốc.",
            "mixture gốc",
            "câu hỏi gốc",
            case.get("qwen_mixture_original"),
        ),
        (
            "Qwen2-Audio",
            "mixture + span-text prompt (from ours)",
            "Qwen2-Audio vẫn nghe audio gốc, nhưng prompt chỉ rõ predicted answer span.",
            "mixture gốc",
            "prompt có predicted answer span",
            case.get("qwen_mixture_span_text"),
        ),
        (
            "Qwen2-Audio",
            "predicted evidence (from ours) + span-text prompt (from ours)",
            "Qwen2-Audio chỉ nghe evidence hệ mình tách ra, kèm prompt chỉ rõ predicted answer span.",
            "predicted evidence của mình",
            "prompt có predicted answer span",
            case.get("qwen_predicted_evidence_eval"),
        ),
        (
            "Phi-4mm",
            "mixture + original question",
            "Phi-4mm nghe toàn bộ audio gốc và trả lời câu hỏi gốc.",
            "mixture gốc",
            "câu hỏi gốc",
            case.get("phi_mixture_original"),
        ),
        (
            "Phi-4mm",
            "mixture + span-text prompt (from ours)",
            "Phi-4mm vẫn nghe audio gốc, nhưng prompt chỉ rõ predicted answer span.",
            "mixture gốc",
            "prompt có predicted answer span",
            case.get("phi_mixture_span_text"),
        ),
        (
            "Phi-4mm",
            "predicted evidence (from ours) + span-text prompt (from ours)",
            "Phi-4mm chỉ nghe evidence hệ mình tách ra, kèm prompt chỉ rõ predicted answer span.",
            "predicted evidence của mình",
            "prompt có predicted answer span",
            case.get("phi_predicted_evidence_eval"),
        ),
    ]:
        if not row:
            rows.append(
                {
                    "evaluator": evaluator,
                    "input": name,
                    "chú thích": note,
                    "audio": audio_desc,
                    "text/prompt": prompt_desc,
                    "prediction": "not run",
                    "correct": "—",
                    "gold_probability": "—",
                }
            )
        else:
            rows.append(
                {
                    "evaluator": evaluator,
                    "input": name,
                    "chú thích": note,
                    "audio": audio_desc,
                    "text/prompt": prompt_desc,
                    "prediction": label_text(row.get("predicted_answer")),
                    "correct": bool(row.get("correct")),
                    "gold_probability": f"{float(row.get('gold_option_probability', 0.0)):.3f}",
                }
            )
    rows.append(
        {
            "evaluator": "ours",
            "input": "ours internal RankCal/pointer",
            "chú thích": "Dự đoán gốc của hệ mình: parser/planner chọn đáp án và span trực tiếp, không qua AF3.",
            "audio": "không gọi AF3",
            "text/prompt": "question parser + RankCal planner",
            "prediction": internal_prediction(case),
            "correct": internal_correct(case),
            "gold_probability": "—",
        }
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("All scene events"):
        st.dataframe(event_table(record), use_container_width=True, hide_index=True)

    with st.expander("Raw metadata"):
        st.json(
            {
                "id": case["id"],
                "pointer_item": pointer,
                "metadata": meta,
                "mixture_original": case.get("mixture_original"),
                "mixture_span_text": case.get("mixture_span_text"),
                "predicted_evidence_eval": case.get("predicted_evidence_eval"),
                "qwen_mixture_original": case.get("qwen_mixture_original"),
                "qwen_mixture_span_text": case.get("qwen_mixture_span_text"),
                "qwen_predicted_evidence_eval": case.get("qwen_predicted_evidence_eval"),
                "phi_mixture_original": case.get("phi_mixture_original"),
                "phi_mixture_span_text": case.get("phi_mixture_span_text"),
                "phi_predicted_evidence_eval": case.get("phi_predicted_evidence_eval"),
            }
        )


if __name__ == "__main__":
    main(sys.argv[1:])
