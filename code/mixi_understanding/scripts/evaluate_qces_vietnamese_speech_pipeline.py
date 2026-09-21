#!/usr/bin/env python3
"""Evaluate Vietnamese speech/event questions and render audible evidence."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import (
    PROJECT_ROOT, _atomic_text, _jsonl, _portable, _render_oracle_evidence,
    _render_temporal_evidence, _resolve, _span_iou,
)


VI_LABEL = {
    "Bark": "tiếng chó sủa", "Meow": "tiếng mèo kêu",
    "Reversing_beeps": "tiếng bíp lùi xe", "Crumpling_and_crinkling": "tiếng vò giấy",
    "Single-lens_reflex_camera": "tiếng máy ảnh", "Chirp_and_tweet": "tiếng chim hót",
    "Engine_starting": "tiếng động cơ khởi động", "Toilet_flush": "tiếng xả nước",
    "Finger_snapping": "tiếng búng tay", "Glass_shatter": "tiếng kính vỡ",
    "Speech": "lời nói tiếng Việt", "Air_horn_and_truck_horn": "tiếng còi xe tải",
}
TRANSCRIPT_OPS = {"vi_speech_content", "vi_speech_content_during_horn", "vi_speech_content_during_event"}
EVENT_OPS = {"vi_event_before_speech", "vi_event_after_speech"}


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _wer(reference: str, hypothesis: str) -> float:
    left, right = _words(reference), _words(hypothesis)
    if not left:
        return 0.0 if not right else 1.0
    previous = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target)))
        previous = current
    return previous[-1] / len(left)


def _parse(text: str) -> str:
    value = text.lower()
    if "thứ hai" in value:
        return "vi_second_speaker_absent"
    if "ngay trước" in value and "bắt đầu nói" in value:
        return "vi_event_before_speech"
    if "ngay sau" in value and "nói xong" in value:
        return "vi_event_after_speech"
    if "còi xe tải" in value and "nói gì" in value:
        return "vi_speech_content_during_horn"
    if "chồng lên lời nói" in value and "nói gì" in value:
        return "vi_speech_content_during_event"
    if "người nói đã nói gì" in value:
        return "vi_speech_content"
    return "unsupported"


def _overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return max(0.0, min(float(a["offset_seconds"]), float(b["offset_seconds"])) - max(float(a["onset_seconds"]), float(b["onset_seconds"])))


def _best_speech(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    speech = [dict(x) for x in events if x["label"] == "Speech"]
    if not speech:
        return None
    return max(speech, key=lambda x: float(x.get("mean_confidence", x["confidence"])) * math.sqrt(max(float(x["offset_seconds"]) - float(x["onset_seconds"]), .01)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asr-model", default=str(Path.home() / ".cache/huggingface/hub/models--vinai--PhoWhisper-tiny/snapshots/cc51d32be916efebde04ff549854fa1741cb5c02"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset, detector_dir, output = args.dataset_dir.resolve(), args.detector_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(dataset / "scenes.jsonl")
    questions = _jsonl(dataset / "questions.jsonl")
    scene_by_id = {x["scene_id"]: x for x in scenes}
    predictions = {}
    for split in ("val", "test"):
        for row in _jsonl(detector_dir / f"detector_predictions_{split}.jsonl"):
            predictions[row["scene_id"]] = row

    device_index = 0 if args.device == "cuda" and torch.cuda.is_available() else -1
    dtype = torch.float16 if device_index >= 0 else torch.float32
    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.asr_model, local_files_only=True, torch_dtype=dtype)
    if device_index >= 0:
        model = model.cuda()
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, device=device_index)

    crop_meta, crop_audio = [], []
    for scene in scenes:
        speech = _best_speech(predictions[scene["scene_id"]]["predicted_events"])
        if speech is None:
            continue
        wave, rate = sf.read(_resolve(scene["mixture_path"]), dtype="float32")
        start = max(0, int((float(speech["onset_seconds"]) - .15) * rate))
        end = min(len(wave), int((float(speech["offset_seconds"]) + .15) * rate))
        crop_meta.append((scene["scene_id"], speech))
        crop_audio.append({"array": wave[start:end], "sampling_rate": rate})
    print(f"PhoWhisper crops={len(crop_audio)}", flush=True)
    hypotheses = asr(crop_audio, batch_size=4, generate_kwargs={"language": "vi", "task": "transcribe"})
    utterance_by_scene = {scene_id: {**speech, "transcript": str(hyp["text"]).strip()} for (scene_id, speech), hyp in zip(crop_meta, hypotheses)}

    rows = []
    qa_metrics = {}
    for split in ("val", "test"):
        correct_all: list[bool] = []
        event_correct: list[bool] = []
        transcript_correct: list[bool] = []
        transcript_wers: list[float] = []
        noev_correct: list[bool] = []
        parser_correct: list[bool] = []
        ious: list[float] = []
        for question in [x for x in questions if x["split"] == split]:
            scene = scene_by_id[question["scene_id"]]
            event_by_id = {x["event_id"]: x for x in scene["events"]}
            detector_events = predictions[scene["scene_id"]]["predicted_events"]
            speech = utterance_by_scene.get(scene["scene_id"])
            sounds = [dict(x) for x in detector_events if x["label"] != "Speech"]
            parsed = _parse(question["question"])
            parser_ok = parsed == question["operation"]
            parser_correct.append(parser_ok)
            answer = "Không có bằng chứng"
            answer_label = None
            selected: list[dict[str, Any]] = []
            anchors: list[dict[str, Any]] = []
            if parsed in TRANSCRIPT_OPS and speech is not None:
                answer = speech["transcript"]
                selected = [speech]
                if parsed == "vi_speech_content_during_event":
                    gold_anchor = event_by_id[question["anchor_event_ids"][0]]["label"]
                    candidates = [x for x in sounds if x["label"] == gold_anchor and _overlap(x, speech) > 0]
                    if candidates:
                        anchors = [max(candidates, key=lambda x: float(x["confidence"]))]
            elif parsed in EVENT_OPS and speech is not None:
                if parsed == "vi_event_before_speech":
                    valid = [x for x in sounds if float(x["offset_seconds"]) <= float(speech["onset_seconds"]) + .15]
                    if valid:
                        chosen = max(valid, key=lambda x: (float(x["offset_seconds"]), float(x["confidence"])))
                else:
                    valid = [x for x in sounds if float(x["onset_seconds"]) >= float(speech["offset_seconds"]) - .15]
                    if valid:
                        chosen = min(valid, key=lambda x: (float(x["onset_seconds"]), -float(x["confidence"])))
                if valid:
                    answer_label = chosen["label"]
                    answer = VI_LABEL.get(answer_label, answer_label.replace("_", " "))
                    selected, anchors = [chosen], [speech]
            elif parsed == "vi_second_speaker_absent":
                anchors = [speech] if speech else []

            if question["operation"] in TRANSCRIPT_OPS:
                error = _wer(question["answer"], answer)
                correct = error <= .25
                transcript_wers.append(error)
                transcript_correct.append(correct)
            elif question["operation"] in EVENT_OPS:
                error = None
                gold_label = event_by_id[question["answer_event_ids"][0]]["label"]
                correct = answer_label == gold_label
                event_correct.append(correct)
            else:
                error = None
                correct = answer == "Không có bằng chứng"
                noev_correct.append(correct)
            correct_all.append(correct)
            pred_spans = [*anchors, *selected]
            gold_events = [event_by_id[x] for x in question["evidence_event_ids"]]
            iou = _span_iou(pred_spans, gold_events, float(scene["duration_seconds"]))
            ious.append(iou)
            pred_path = oracle_path = ""
            if split == "test":
                pp = output / "evidence/predicted" / f"{question['question_id']}.flac"
                op = output / "evidence/oracle" / f"{question['question_id']}.flac"
                _render_temporal_evidence(_resolve(scene["mixture_path"]), pred_spans, pp)
                _render_oracle_evidence(scene, question, op)
                pred_path, oracle_path = _portable(pp), _portable(op)
            rows.append({
                "format": "qces_vietnamese_speech_pipeline_result_v1", "question_id": question["question_id"],
                "scene_id": scene["scene_id"], "split": split, "question": question["question"],
                "operation": question["operation"], "parsed_operation": parsed, "parser_correct": parser_ok,
                "gold_answer": question["answer"], "predicted_answer": answer, "correct": correct,
                "transcript_wer_↓": error, "evidence_iou_↑": iou, "mixture_path": scene["mixture_path"],
                "predicted_evidence_path": pred_path, "oracle_evidence_path": oracle_path,
                "predicted_evidence_spans": pred_spans,
                "predicted_utterances": [speech] if speech else [], "predicted_events": detector_events,
            })
        qa_metrics[split] = {
            "questions": len(correct_all), "overall_accuracy_↑": float(np.mean(correct_all)),
            "event_answer_accuracy_↑": float(np.mean(event_correct)),
            "transcript_accuracy_at_wer_0.25_↑": float(np.mean(transcript_correct)),
            "mean_transcript_wer_↓": float(np.mean(transcript_wers)),
            "no_evidence_accuracy_↑": float(np.mean(noev_correct)),
            "question_parser_accuracy_↑": float(np.mean(parser_correct)), "mean_evidence_iou_↑": float(np.mean(ious)),
        }
    result_path = output / "pipeline_results.jsonl"
    _atomic_text(result_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows))
    detector_receipt = json.loads((detector_dir / "receipt.json").read_text())
    receipt = {
        "format": "qces_vietnamese_speech_pipeline_receipt_v1", "complete": True,
        "model": {"event_and_speech_detector": "frozen BEATs-Strong + existing 11-class temporal head",
                  "asr": "VinAI PhoWhisper-tiny", "question_parser": "six deterministic Vietnamese demo templates",
                  "evidence": "mixture masked by predicted temporal spans"},
        "detector": {split: detector_receipt["splits"][split] for split in ("val", "test")},
        "qa": qa_metrics, "result_manifest": _portable(result_path),
        "result_manifest_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(qa_metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
