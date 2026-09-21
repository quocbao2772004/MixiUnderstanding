#!/usr/bin/env python3
"""Evaluate gender, speaker-order, event QA, and evidence for the multi-speaker set."""

from __future__ import annotations

import collections
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from mixi_understanding.scripts.train_qces_multispeaker_speech_detector import (
    LABELS,
    _iou,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DIFFICULTIES = (("easy_overlap", 6.0), ("medium_overlap", 0.0), ("hard_overlap", -6.0))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{__import__('os').getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _portable(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return max(0.0, min(float(left["offset_seconds"]), float(right["offset_seconds"])) - max(float(left["onset_seconds"]), float(right["onset_seconds"])))


def _center(event: Mapping[str, Any]) -> float:
    return 0.5 * (float(event["onset_seconds"]) + float(event["offset_seconds"]))


def _speaker_gender(label: str) -> str | None:
    if label == "Speech_male":
        return "male"
    if label == "Speech_female":
        return "female"
    return None


def _execute(operation: str, predicted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    speech = [dict(event) for event in predicted if _speaker_gender(str(event["label"]))]
    speech.sort(key=lambda event: (float(event["onset_seconds"]), -float(event["confidence"])))
    sounds = [dict(event) for event in predicted if not _speaker_gender(str(event["label"]))]
    result: dict[str, Any] = {"answer": "No evidence", "selected": [], "anchors": []}
    if operation == "third_speaker_absent":
        result["answer"] = "No evidence" if len(speech) < 3 else "evidence"
        result["selected"] = speech
        return result
    if len(speech) < 2:
        return result
    first, second = speech[0], speech[1]
    if operation == "speaker_first":
        result.update(answer=_speaker_gender(first["label"]), selected=[first], anchors=[first, second])
    elif operation == "speaker_second":
        result.update(answer=_speaker_gender(second["label"]), selected=[second], anchors=[first, second])
    elif operation == "event_before_first_speech":
        candidates = [event for event in sounds if _center(event) < float(first["onset_seconds"])]
        if candidates:
            chosen = max(candidates, key=lambda event: (_center(event), float(event["confidence"])))
            result.update(answer=chosen["label"], selected=[chosen], anchors=[first])
    elif operation == "event_between_speakers":
        candidates = [event for event in sounds if _center(event) > float(first["offset_seconds"]) and _center(event) < float(second["onset_seconds"])]
        if candidates:
            chosen = min(candidates, key=lambda event: (_center(event), -float(event["confidence"])))
            result.update(answer=chosen["label"], selected=[chosen], anchors=[first, second])
    elif operation == "event_overlap_second_speech":
        candidates = [(event, _overlap(event, second)) for event in sounds]
        candidates = [(event, value) for event, value in candidates if value > 0]
        if candidates:
            chosen = max(candidates, key=lambda item: (item[1], float(item[0]["confidence"])))[0]
            result.update(answer=chosen["label"], selected=[chosen], anchors=[second])
    elif operation == "event_after_second_speech":
        candidates = [event for event in sounds if _center(event) > float(second["offset_seconds"])]
        if candidates:
            chosen = min(candidates, key=lambda event: (_center(event) - float(second["offset_seconds"]), -float(event["confidence"])))
            result.update(answer=chosen["label"], selected=[chosen], anchors=[second])
    return result


def _span_iou(predicted: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]], duration: float) -> float:
    frames = max(1, int(math.ceil(duration * 100)))
    left = np.zeros(frames, dtype=bool)
    right = np.zeros(frames, dtype=bool)
    for values, target in ((predicted, left), (gold, right)):
        for event in values:
            start = max(0, min(frames, int(math.floor(float(event["onset_seconds"]) * 100))))
            end = max(start, min(frames, int(math.ceil(float(event["offset_seconds"]) * 100))))
            target[start:end] = True
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 1.0


def _render_predicted_evidence(mixture_path: Path, spans: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    waveform, sample_rate = sf.read(mixture_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    mask = np.zeros(len(waveform), dtype=np.float32)
    for span in spans:
        start = max(0, min(len(mask), int(float(span["onset_seconds"]) * sample_rate)))
        end = max(start, min(len(mask), int(float(span["offset_seconds"]) * sample_rate)))
        mask[start:end] = 1.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform * mask, sample_rate, format="FLAC", subtype="PCM_16")


def _render_oracle_evidence(scene: Mapping[str, Any], event_ids: Sequence[str], output_path: Path) -> None:
    event_by_id = {str(event["event_id"]): event for event in scene["events"]}
    mixture, sample_rate = sf.read(_resolve(str(scene["mixture_path"])), dtype="float32")
    evidence = np.zeros_like(mixture, dtype=np.float32)
    for event_id in event_ids:
        stem, stem_rate = sf.read(_resolve(str(event_by_id[str(event_id)]["stem_path"])), dtype="float32")
        if stem_rate != sample_rate:
            raise RuntimeError(f"oracle stem sample-rate mismatch: {event_id}")
        evidence += stem
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, evidence, sample_rate, format="FLAC", subtype="PCM_16")


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = [bool(row["correct"]) for row in rows]
    by_operation: dict[str, list[bool]] = collections.defaultdict(list)
    for row in rows:
        by_operation[str(row["operation"])].append(bool(row["correct"]))
    return {
        "questions": len(rows),
        "overall_accuracy_↑": float(np.mean(correct)) if correct else 0.0,
        "speaker_gender_accuracy_↑": float(np.mean([row["correct"] for row in rows if str(row["operation"]).startswith("speaker_")])) if any(str(row["operation"]).startswith("speaker_") for row in rows) else 0.0,
        "event_answer_accuracy_↑": float(np.mean([row["correct"] for row in rows if str(row["operation"]).startswith("event_")])) if any(str(row["operation"]).startswith("event_") for row in rows) else 0.0,
        "no_evidence_accuracy_↑": float(np.mean([row["correct"] for row in rows if row["operation"] == "third_speaker_absent"])) if any(row["operation"] == "third_speaker_absent" for row in rows) else 0.0,
        "mean_evidence_iou_↑": float(np.mean([row["evidence_iou_↑"] for row in rows])) if rows else 0.0,
        "by_operation": {operation: {"items": len(values), "accuracy_↑": float(np.mean(values))} for operation, values in sorted(by_operation.items())},
    }


def _detector_metrics(scenes: Sequence[Mapping[str, Any]], prediction_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = 0
    matched: list[float] = []
    per_label = {label: collections.Counter() for label in LABELS}
    for scene in scenes:
        predicted = list(prediction_rows.get(str(scene["scene_id"]), {}).get("predicted_events", []))
        gold = [event for event in scene["events"] if event["label"] in LABELS]
        used: set[int] = set()
        for candidate in sorted(predicted, key=lambda event: float(event["confidence"]), reverse=True):
            choices = [(index, _iou(candidate, event)) for index, event in enumerate(gold) if index not in used and event["label"] == candidate["label"]]
            best_index, best_iou = max(choices, key=lambda value: value[1], default=(-1, 0.0))
            if best_iou >= 0.30:
                used.add(best_index)
                tp += 1
                per_label[candidate["label"]]["tp"] += 1
                matched.append(best_iou)
            else:
                fp += 1
                per_label[candidate["label"]]["fp"] += 1
        for index, event in enumerate(gold):
            if index not in used:
                fn += 1
                per_label[event["label"]]["fn"] += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "event_precision_↑": precision,
        "event_recall_↑": recall,
        "event_f1_↑": 2 * precision * recall / max(precision + recall, 1e-12),
        "matched_mean_iou_↑": float(np.mean(matched)) if matched else 0.0,
        "per_label": {label: {**counts, "recall_↑": counts["tp"] / max(counts["tp"] + counts["fn"], 1)} for label, counts in per_label.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_multispeaker_speech_event_v1")
    parser.add_argument("--detector-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_multispeaker_speech_detector_v1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_multispeaker_speech_pipeline_v1")
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    detector = args.detector_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(dataset / "scenes.jsonl")
    questions = _jsonl(dataset / "questions.jsonl")
    scene_by_id = {str(scene["scene_id"]): scene for scene in scenes}
    prediction_rows: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        for row in _jsonl(detector / f"detector_predictions_{split}.jsonl"):
            prediction_rows[str(row["scene_id"])] = row
    result_rows: list[dict[str, Any]] = []
    for question in questions:
        if str(question["split"]) == "train":
            continue
        scene = scene_by_id[str(question["scene_id"])]
        prediction = prediction_rows[str(question["scene_id"])]
        execution = _execute(str(question["operation"]), prediction["predicted_events"])
        event_by_id = {str(event["event_id"]): event for event in scene["events"]}
        gold = [event_by_id[str(event_id)] for event_id in question["evidence_event_ids"]]
        predicted_spans = [*execution["anchors"], *execution["selected"]]
        predicted_evidence_path = output / "evidence/predicted" / f"{question['question_id']}.flac"
        oracle_evidence_path = output / "evidence/oracle" / f"{question['question_id']}.flac"
        _render_predicted_evidence(_resolve(str(scene["mixture_path"])), predicted_spans, predicted_evidence_path)
        _render_oracle_evidence(scene, question["evidence_event_ids"], oracle_evidence_path)
        if question["operation"].startswith("speaker_"):
            correct = execution["answer"] == question["answer"]
        elif question["operation"] == "third_speaker_absent":
            correct = execution["answer"] == "No evidence"
        else:
            correct = execution["answer"] == question["answer"]
        result_rows.append({
            "format": "qces_multispeaker_speech_pipeline_result_v1",
            "question_id": question["question_id"],
            "scene_id": question["scene_id"],
            "split": question["split"],
            "difficulty": scene["difficulty"],
            "operation": question["operation"],
            "gold_answer": question["answer"],
            "predicted_answer": execution["answer"],
            "correct": bool(correct),
            "evidence_iou_↑": _span_iou(predicted_spans, gold, float(scene["duration_seconds"])),
            "predicted_evidence_path": _portable(predicted_evidence_path),
            "oracle_evidence_path": _portable(oracle_evidence_path),
            "predicted_events": prediction["predicted_events"],
        })
    split_metrics: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in result_rows if row["split"] == split]
        split_metrics[split] = _metrics(split_rows)
        for difficulty, _ in DIFFICULTIES:
            difficulty_rows = [row for row in split_rows if row["difficulty"] == difficulty]
            if difficulty_rows:
                split_metrics[split][difficulty] = _metrics(difficulty_rows)
        split_metrics[split]["detector"] = _detector_metrics([scene for scene in scenes if scene["split"] == split], prediction_rows) if split != "train" else None
    result_path = output / "pipeline_results.jsonl"
    _atomic_text(result_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows))
    receipt = {
        "format": "qces_multispeaker_speech_pipeline_receipt_v1",
        "complete": True,
        "model": "frozen BEATs-Strong + gender-aware temporal head",
        "questions": "gender, speaker order, event-before/after/between/overlap, no-evidence",
        "labels": list(LABELS),
        "metrics": split_metrics,
        "result_manifest": str(result_path.relative_to(PROJECT_ROOT)),
        "result_manifest_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(split_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
