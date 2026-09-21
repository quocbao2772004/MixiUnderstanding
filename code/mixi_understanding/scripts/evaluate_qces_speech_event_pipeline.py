#!/usr/bin/env python3
"""Run speech-aware QA, ASR, and evidence rendering for QCES speech-event v1."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


EVENT_OPERATIONS = {
    "event_before_speech",
    "event_after_speech",
    "event_during_speech",
    "event_during_speech_ordinal",
    "event_during_second_speech",
    "event_between_speech",
    "first_event",
}
TRANSCRIPT_OPERATIONS = {"speech_content_ordinal", "speech_after_quote"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _portable(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _normalize_text(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower().replace("'", ""))


def _word_error_rate(reference: str, hypothesis: str) -> float:
    left, right = _normalize_text(reference), _normalize_text(hypothesis)
    if not left:
        return 0.0 if not right else 1.0
    previous = list(range(len(right) + 1))
    for row_index, source in enumerate(left, start=1):
        current = [row_index]
        for column_index, target in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (source != target),
                )
            )
        previous = current
    return previous[-1] / len(left)


def _parse_question(text: str) -> str:
    value = text.lower()
    if "immediately before" in value and "starts speaking" in value:
        return "event_before_speech"
    if "sound occurs" in value and "after" in value and "utterance" in value:
        return "event_after_speech"
    if "sound starts first while" in value or "sound starts second while" in value:
        return "event_during_speech_ordinal"
    if "sound overlaps" in value and "second utterance" in value:
        return "event_during_second_speech"
    if "sound overlaps" in value:
        return "event_during_speech"
    if "sound occurs between" in value:
        return "event_between_speech"
    if "what did the woman say after" in value:
        return "speech_after_quote"
    if "what did the woman say first" in value or "what did the woman say second" in value:
        return "speech_content_ordinal"
    if "what did the man say" in value:
        return "speaker_absent"
    if "first sound event" in value:
        return "first_event"
    return "unsupported"


def _select_speech(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = [dict(event) for event in events if event["label"] == "Speech"]
    candidates.sort(
        key=lambda event: (
            float(event["mean_confidence"])
            * math.sqrt(max(float(event["offset_seconds"]) - float(event["onset_seconds"]), 0.01)),
            float(event["confidence"]),
        ),
        reverse=True,
    )
    return sorted(candidates[:2], key=lambda event: float(event["onset_seconds"]))


def _event_center(event: Mapping[str, Any]) -> float:
    return 0.5 * (float(event["onset_seconds"]) + float(event["offset_seconds"]))


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return max(
        0.0,
        min(float(left["offset_seconds"]), float(right["offset_seconds"]))
        - max(float(left["onset_seconds"]), float(right["onset_seconds"])),
    )


def _execute(
    operation: str,
    question: str,
    detector_events: Sequence[Mapping[str, Any]],
    utterances: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    speech = list(utterances)
    sounds = [dict(event) for event in detector_events if event["label"] != "Speech"]
    result = {"answer": "No evidence", "answer_label": None, "selected": [], "anchors": []}
    if operation == "speaker_absent":
        result["selected"] = list(speech)
        return result
    if operation == "first_event":
        if sounds:
            chosen = min(sounds, key=lambda event: (_event_center(event), -float(event["confidence"])))
            result.update(answer=chosen["label"], answer_label=chosen["label"], selected=[chosen])
        return result
    if operation in TRANSCRIPT_OPERATIONS:
        if operation == "speech_after_quote" or "second" in question.lower():
            index = 1
        else:
            index = 0
        if len(speech) > index:
            chosen = speech[index]
            anchors = [speech[0]] if operation == "speech_after_quote" and len(speech) > 1 else []
            result.update(answer=chosen.get("transcript") or "", selected=[chosen], anchors=anchors)
        return result
    if not speech:
        return result
    if operation == "event_before_speech":
        anchor = speech[0]
        valid = [event for event in sounds if _event_center(event) < float(anchor["onset_seconds"])]
        if valid:
            chosen = max(valid, key=lambda event: (_event_center(event), float(event["confidence"])))
            result.update(answer=chosen["label"], answer_label=chosen["label"], selected=[chosen], anchors=[anchor])
    elif operation == "event_after_speech":
        anchor_index = 1 if "second" in question.lower() and len(speech) > 1 else 0
        anchor = speech[anchor_index]
        valid = [event for event in sounds if _event_center(event) > float(anchor["offset_seconds"])]
        if valid:
            chosen = min(
                valid,
                key=lambda event: (
                    _event_center(event) - float(anchor["offset_seconds"]),
                    -float(event["confidence"]),
                ),
            )
            result.update(answer=chosen["label"], answer_label=chosen["label"], selected=[chosen], anchors=[anchor])
    elif operation == "event_during_speech":
        anchor = speech[0]
        candidates = []
        anchor_duration = max(float(anchor["offset_seconds"]) - float(anchor["onset_seconds"]), 1e-6)
        for event in sounds:
            overlap = _overlap(event, anchor)
            event_duration = max(float(event["offset_seconds"]) - float(event["onset_seconds"]), 1e-6)
            relative = (_event_center(event) - float(anchor["onset_seconds"])) / anchor_duration
            interior = 1.0 if 0.15 <= relative <= 0.85 else 0.0
            score = overlap / event_duration + 0.5 * interior + 0.1 * float(event["confidence"])
            if overlap > 0:
                candidates.append((score, event))
        if candidates:
            chosen = max(candidates, key=lambda value: value[0])[1]
            result.update(answer=chosen["label"], answer_label=chosen["label"], selected=[chosen], anchors=[anchor])
    elif operation == "event_during_speech_ordinal":
        anchor = speech[0]
        valid = [
            event
            for event in sounds
            if _overlap(event, anchor) > 0
            and float(anchor["onset_seconds"]) < _event_center(event) < float(anchor["offset_seconds"])
        ]
        valid.sort(key=lambda event: (_event_center(event), -float(event["confidence"])))
        ordinal = 1 if "starts second" in question.lower() else 0
        if len(valid) > ordinal:
            chosen = valid[ordinal]
            result.update(answer=chosen["label"], answer_label=chosen["label"], selected=[chosen], anchors=[anchor])
    elif operation == "event_during_second_speech" and len(speech) > 1:
        anchor = speech[1]
        valid = [
            event
            for event in sounds
            if _overlap(event, anchor) > 0
            and float(anchor["onset_seconds"]) < _event_center(event) < float(anchor["offset_seconds"])
        ]
        if valid:
            midpoint = _event_center(anchor)
            chosen = min(
                valid,
                key=lambda event: (
                    abs(_event_center(event) - midpoint),
                    -float(event["confidence"]),
                ),
            )
            result.update(answer=chosen["label"], answer_label=chosen["label"], selected=[chosen], anchors=[anchor])
    elif operation == "event_between_speech" and len(speech) > 1:
        left, right = speech[:2]
        valid = [
            event for event in sounds
            if _event_center(event) > float(left["offset_seconds"])
            and _event_center(event) < float(right["onset_seconds"])
        ]
        if valid:
            midpoint = 0.5 * (float(left["offset_seconds"]) + float(right["onset_seconds"]))
            chosen = min(valid, key=lambda event: (abs(_event_center(event) - midpoint), -float(event["confidence"])))
            result.update(answer=chosen["label"], answer_label=chosen["label"], selected=[chosen], anchors=[left, right])
    return result


def _render_temporal_evidence(
    mixture_path: Path, spans: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    waveform, sample_rate = sf.read(mixture_path, dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1, dtype=np.float32)
    mask = np.zeros(len(mono), dtype=np.float32)
    fade = max(1, int(round(0.02 * sample_rate)))
    for span in spans:
        start = max(0, min(len(mask), int(math.floor(float(span["onset_seconds"]) * sample_rate))))
        end = max(start, min(len(mask), int(math.ceil(float(span["offset_seconds"]) * sample_rate))))
        if end <= start:
            continue
        mask[start:end] = 1.0
        width = min(fade, (end - start) // 2)
        if width:
            ramp = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
            mask[start : start + width] = np.maximum(mask[start : start + width], ramp)
            mask[end - width : end] = np.maximum(mask[end - width : end], ramp[::-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mono * mask, sample_rate, format="FLAC", subtype="PCM_16")


def _render_oracle_evidence(scene: Mapping[str, Any], question: Mapping[str, Any], output_path: Path) -> None:
    event_by_id = {event["event_id"]: event for event in scene["events"]}
    mixture, sample_rate = sf.read(_resolve(str(scene["mixture_path"])), dtype="float32")
    evidence = np.zeros_like(mixture)
    for event_id in question["evidence_event_ids"]:
        event = event_by_id[event_id]
        stem, stem_rate = sf.read(_resolve(str(event["stem_path"])), dtype="float32")
        if stem_rate != sample_rate or len(stem) != len(evidence):
            raise RuntimeError(f"oracle stem grid mismatch: {event_id}")
        evidence += stem
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, evidence, sample_rate, format="FLAC", subtype="PCM_16")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=PROJECT_ROOT / "data/qces_speech_event_v1_smoke100",
    )
    parser.add_argument(
        "--detector-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_speech_event_v1_detector",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_speech_event_v1_pipeline",
    )
    parser.add_argument("--asr-model", default="openai/whisper-tiny.en")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset_dir, detector_dir, output_dir = (
        args.dataset_dir.resolve(), args.detector_dir.resolve(), args.output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(dataset_dir / "scenes.jsonl")
    questions = _jsonl(dataset_dir / "questions.jsonl")
    scene_by_id = {scene["scene_id"]: scene for scene in scenes}
    predictions: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        for row in _jsonl(detector_dir / f"detector_predictions_{split}.jsonl"):
            if row["scene_id"] in scene_by_id:
                predictions[row["scene_id"]] = row

    device_index = 0 if args.device == "cuda" and torch.cuda.is_available() else -1
    dtype = torch.float16 if device_index >= 0 else torch.float32
    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.asr_model, local_files_only=True, dtype=dtype
    )
    if device_index >= 0:
        model = model.cuda()
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device_index,
    )

    crops: list[dict[str, Any]] = []
    crop_audio: list[dict[str, Any]] = []
    for scene_id, detector in predictions.items():
        scene = scene_by_id[scene_id]
        waveform, sample_rate = sf.read(_resolve(scene["mixture_path"]), dtype="float32")
        for ordinal, event in enumerate(_select_speech(detector["predicted_events"]), start=1):
            start = max(0, int(math.floor((float(event["onset_seconds"]) - 0.10) * sample_rate)))
            end = min(len(waveform), int(math.ceil((float(event["offset_seconds"]) + 0.10) * sample_rate)))
            crop_audio.append({"array": waveform[start:end], "sampling_rate": sample_rate})
            crops.append({"scene_id": scene_id, "ordinal": ordinal, **event})
    print(f"ASR crops={len(crops)}", flush=True)
    hypotheses = asr(crop_audio, batch_size=4)
    utterances_by_scene: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for crop, hypothesis in zip(crops, hypotheses):
        utterances_by_scene[crop["scene_id"]].append(
            {**crop, "transcript": str(hypothesis["text"]).strip()}
        )
    for values in utterances_by_scene.values():
        values.sort(key=lambda event: float(event["onset_seconds"]))

    result_rows: list[dict[str, Any]] = []
    split_metrics: dict[str, Any] = {}
    for split in ("val", "test"):
        event_correct: list[bool] = []
        transcript_correct: list[bool] = []
        transcript_wers: list[float] = []
        noev_correct: list[bool] = []
        parser_correct: list[bool] = []
        evidence_ious: list[float] = []
        split_questions = [question for question in questions if question["split"] == split]
        for question in split_questions:
            scene = scene_by_id[question["scene_id"]]
            detector = predictions[question["scene_id"]]
            utterances = utterances_by_scene.get(question["scene_id"], [])
            parsed = _parse_question(question["question"])
            parser_ok = parsed == question["operation"]
            parser_correct.append(parser_ok)
            execution = _execute(parsed, question["question"], detector["predicted_events"], utterances)
            event_by_id = {event["event_id"]: event for event in scene["events"]}
            gold_events = [event_by_id[event_id] for event_id in question["evidence_event_ids"]]
            predicted_spans = [*execution["anchors"], *execution["selected"]]
            evidence_iou = _span_iou(predicted_spans, gold_events, float(scene["duration_seconds"]))
            evidence_ious.append(evidence_iou)
            wer: float | None = None
            if question["operation"] in EVENT_OPERATIONS:
                gold_label = event_by_id[question["answer_event_ids"][0]]["label"]
                correct = execution["answer_label"] == gold_label
                event_correct.append(correct)
            elif question["operation"] in TRANSCRIPT_OPERATIONS:
                wer = _word_error_rate(question["answer"], execution["answer"])
                correct = wer <= 0.25
                transcript_wers.append(wer)
                transcript_correct.append(correct)
            else:
                correct = execution["answer"] == "No evidence"
                noev_correct.append(correct)
            predicted_evidence_path = ""
            oracle_evidence_path = ""
            if split == "test":
                predicted_path = output_dir / "evidence/predicted" / f"{question['question_id']}.flac"
                oracle_path = output_dir / "evidence/oracle" / f"{question['question_id']}.flac"
                _render_temporal_evidence(_resolve(scene["mixture_path"]), predicted_spans, predicted_path)
                _render_oracle_evidence(scene, question, oracle_path)
                predicted_evidence_path = _portable(predicted_path)
                oracle_evidence_path = _portable(oracle_path)
            result_rows.append(
                {
                    "format": "qces_speech_event_pipeline_result_v1",
                    "question_id": question["question_id"],
                    "scene_id": question["scene_id"],
                    "split": split,
                    "question": question["question"],
                    "operation": question["operation"],
                    "parsed_operation": parsed,
                    "parser_correct": parser_ok,
                    "gold_answer": question["answer"],
                    "predicted_answer": execution["answer"],
                    "correct": correct,
                    "transcript_wer_↓": wer,
                    "evidence_iou_↑": evidence_iou,
                    "mixture_path": scene["mixture_path"],
                    "predicted_evidence_path": predicted_evidence_path,
                    "oracle_evidence_path": oracle_evidence_path,
                    "predicted_evidence_spans": predicted_spans,
                    "predicted_utterances": utterances,
                    "predicted_events": detector["predicted_events"],
                }
            )
        all_correct = [*event_correct, *transcript_correct, *noev_correct]
        split_metrics[split] = {
            "questions": len(split_questions),
            "overall_accuracy_↑": sum(all_correct) / max(len(all_correct), 1),
            "event_answer_accuracy_↑": sum(event_correct) / max(len(event_correct), 1),
            "transcript_accuracy_at_wer_0.25_↑": sum(transcript_correct) / max(len(transcript_correct), 1),
            "mean_transcript_wer_↓": float(np.mean(transcript_wers)) if transcript_wers else None,
            "no_evidence_accuracy_↑": sum(noev_correct) / max(len(noev_correct), 1),
            "question_parser_accuracy_↑": sum(parser_correct) / max(len(parser_correct), 1),
            "mean_evidence_iou_↑": float(np.mean(evidence_ious)),
        }

    result_path = output_dir / "pipeline_results.jsonl"
    _atomic_text(result_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows))
    detector_receipt = json.loads((detector_dir / "receipt.json").read_text(encoding="utf-8"))
    receipt = {
        "format": "qces_speech_event_pipeline_receipt_v1",
        "complete": True,
        "model": {
            "event_and_speech_detector": "frozen PretrainedSED BEATs-Strong + trained 11-class temporal head",
            "asr": args.asr_model,
            "question_parser": "deterministic parser for the ten benchmark templates",
            "evidence": "mixture masked by predicted temporal spans",
        },
        "detector": {"val": detector_receipt["val"], "test": detector_receipt["test"]},
        "qa": split_metrics,
        "result_manifest": _portable(result_path),
        "result_manifest_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output_dir / "receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(split_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
