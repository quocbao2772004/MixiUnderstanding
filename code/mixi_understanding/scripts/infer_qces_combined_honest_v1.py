#!/usr/bin/env python3
"""Run the combined automotive demo without target-scene annotations at inference.

The inference path is deliberately separated from evaluation:

* ``detector`` reads only mixture waveforms and uses the public, unmodified
  BEATs-Strong AudioSet checkpoint with a fixed 0.5 threshold.
* ``asr`` runs PhoWhisper-large on the complete mixture and obtains word
  timestamps.  It never crops with the annotated speech onset/offset.
* ``evaluate`` is the only stage allowed to read event annotations and speech
  transcripts.  It joins the two prediction files, computes metrics, and emits
  the frozen payload consumed by Streamlit.

This is an honest zero-shot diagnostic.  It is not advertised as a strict
dataset-disjoint benchmark because the environmental sources are AudioSet
derived and the public detector was pretrained on AudioSet.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_SED_ROOT = PROJECT_ROOT / "code/baseline/PretrainedSED"
for value in (PROJECT_ROOT / "code", PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

from mixi_understanding.audioqa_qwen_contract import (
    ParsedAudioQuestion,
    answer_event_graph_program,
)


DATASET = PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_realistic_v1"
OUTPUT = PROJECT_ROOT / "outputs/qces_combined_honest_zero_shot_v1"
MODEL_ID = "vinai/PhoWhisper-large"
DETECTOR_THRESHOLD = 0.5
MEDIAN_FRAMES = 9
FRAME_SECONDS = 0.04

RAW_TO_LABEL = {
    "Accelerating, revving, vroom": "Accelerating_and_revving",
    "Air horn, truck horn": "Air_horn_and_truck_horn",
    "Bark": "Bark",
    "Caterwaul": "Caterwaul",
    "Clapping": "Clapping",
    "Conversation": "Conversation",
    "Crying, sobbing": "Crying_and_sobbing",
    "Engine starting": "Engine_starting",
    "Giggle": "Giggle",
    "Heavy engine (low frequency)": "Heavy_engine_(low_frequency)",
    "Howl": "Howl",
    "Knock": "Knock",
    "Laughter": "Laughter",
    "Meow": "Meow",
    "Medium engine (mid frequency)": "Medium_engine_(mid_frequency)",
    "Police car (siren)": "Police_car_(siren)",
    "Purr": "Purr",
    "Reversing beeps": "Reversing_beeps",
    "Shout": "Shout",
    "Slam": "Slam",
    "Thump, thud": "Thump_and_thud",
    "Traffic noise, roadway noise": "Traffic_noise_and_roadway_noise",
    "Whimper (dog)": "Whimper_(dog)",
}
DISPLAY = {
    "Accelerating_and_revving": "tiếng động cơ tăng tốc/rồ ga",
    "Air_horn_and_truck_horn": "tiếng còi xe tải",
    "Bark": "tiếng chó sủa",
    "Caterwaul": "tiếng mèo gào",
    "Clapping": "tiếng vỗ tay",
    "Conversation": "tiếng trò chuyện",
    "Crying_and_sobbing": "tiếng khóc/nức nở",
    "Engine_starting": "tiếng động cơ khởi động",
    "Giggle": "tiếng cười khúc khích",
    "Heavy_engine_(low_frequency)": "tiếng động cơ hạng nặng",
    "Howl": "tiếng chó tru",
    "Knock": "tiếng gõ/đập bàn",
    "Laughter": "tiếng cười",
    "Meow": "tiếng mèo kêu",
    "Medium_engine_(mid_frequency)": "tiếng động cơ xe tần số trung",
    "Police_car_(siren)": "tiếng còi xe cảnh sát",
    "Purr": "tiếng mèo rừ",
    "Reversing_beeps": "tiếng bíp lùi xe",
    "Shout": "tiếng hét",
    "Slam": "tiếng đóng/đập mạnh",
    "Thump_and_thud": "tiếng va đập trầm/đập ghế",
    "Traffic_noise_and_roadway_noise": "tiếng giao thông trên đường",
    "Whimper_(dog)": "tiếng chó rên",
    "Speech": "tiếng người nói",
}
SPEECH_RAW = (
    "Speech",
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Child speech, kid speaking",
)
GENDER_RAW = {
    "female": "Female speech, woman speaking",
    "male": "Male speech, man speaking",
    "child": "Child speech, kid speaking",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_text(path: Path, value: str) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio(path: Path, sample_rate: int = 16_000) -> tuple[np.ndarray, int]:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if int(source_rate) != sample_rate:
        divisor = math.gcd(int(source_rate), sample_rate)
        waveform = resample_poly(waveform, sample_rate // divisor, int(source_rate) // divisor)
    return np.asarray(waveform, dtype=np.float32), sample_rate


def contiguous_segments(
    values: np.ndarray,
    *,
    threshold: float,
    valid_frames: int,
    merge_gap_seconds: float,
    minimum_seconds: float,
) -> list[tuple[int, int]]:
    flags = (values[:valid_frames] >= threshold).tolist() + [False]
    raw: list[tuple[int, int]] = []
    start: int | None = None
    for frame, active in enumerate(flags):
        if active and start is None:
            start = frame
        elif not active and start is not None:
            raw.append((start, frame))
            start = None
    gap = int(math.floor(merge_gap_seconds / FRAME_SECONDS + 1e-8))
    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    minimum = max(1, int(math.ceil(minimum_seconds / FRAME_SECONDS - 1e-8)))
    return [(start, end) for start, end in merged if end - start >= minimum]


def detector_stage(dataset: Path, output: Path, device_name: str) -> None:
    # Imported lazily so the ASR-only stage does not initialize BEATs.
    import models.prediction_wrapper as prediction_wrapper
    from data_util.audioset_classes import as_strong_train_classes
    from models.beats.BEATs_wrapper import BEATsWrapper
    from models.prediction_wrapper import PredictionsWrapper

    prediction_wrapper.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    scenes = read_jsonl(dataset / "scenes.jsonl")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
    model.eval().to(device)
    label_to_id = {label: index for index, label in enumerate(as_strong_train_classes)}
    required = set(RAW_TO_LABEL) | set(SPEECH_RAW)
    missing = sorted(required - set(label_to_id))
    if missing:
        raise RuntimeError(f"public detector is missing required labels: {missing}")
    rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        waveform, sample_rate = load_audio(resolve(str(scene["mixture_path"])))
        duration = len(waveform) / sample_rate
        audio = torch.from_numpy(waveform).unsqueeze(0)
        target_samples = 10 * sample_rate
        audio = audio[:, :target_samples]
        if audio.shape[1] < target_samples:
            audio = F.pad(audio, (0, target_samples - audio.shape[1]))
        with torch.inference_mode():
            mel = model.mel_forward(audio.to(device))
            logits, _ = model(mel)
            probabilities = logits.sigmoid()[0].transpose(0, 1).float().cpu().numpy()
        probabilities = median_filter(probabilities, size=(MEDIAN_FRAMES, 1), mode="nearest")
        valid_frames = min(probabilities.shape[0], int(math.ceil(duration / FRAME_SECONDS)))
        events: list[dict[str, Any]] = []
        for raw_label, label in RAW_TO_LABEL.items():
            values = probabilities[:, label_to_id[raw_label]]
            for start, end in contiguous_segments(
                values,
                threshold=DETECTOR_THRESHOLD,
                valid_frames=valid_frames,
                merge_gap_seconds=0.16,
                minimum_seconds=0.08,
            ):
                events.append(
                    {
                        "label": label,
                        "display_label": DISPLAY[label],
                        "start_seconds": start * FRAME_SECONDS,
                        "end_seconds": min(duration, end * FRAME_SECONDS),
                        "confidence": float(values[start:end].max()),
                        "mean_confidence": float(values[start:end].mean()),
                        "source": "public_beats_strong_zero_shot",
                    }
                )
        speech_values = np.max(
            np.stack([probabilities[:, label_to_id[label]] for label in SPEECH_RAW], axis=1),
            axis=1,
        )
        speech_segments = contiguous_segments(
            speech_values,
            threshold=DETECTOR_THRESHOLD,
            valid_frames=valid_frames,
            merge_gap_seconds=0.30,
            minimum_seconds=0.30,
        )
        for start, end in speech_segments:
            events.append(
                {
                    "label": "Speech",
                    "display_label": DISPLAY["Speech"],
                    "start_seconds": start * FRAME_SECONDS,
                    "end_seconds": min(duration, end * FRAME_SECONDS),
                    "confidence": float(speech_values[start:end].max()),
                    "mean_confidence": float(speech_values[start:end].mean()),
                    "source": "public_beats_strong_zero_shot",
                }
            )
        active_frames = np.flatnonzero(speech_values[:valid_frames] >= DETECTOR_THRESHOLD)
        speaker_scores = {
            speaker: float(
                probabilities[
                    active_frames if len(active_frames) else slice(0, valid_frames),
                    label_to_id[raw_label],
                ].mean()
            )
            for speaker, raw_label in GENDER_RAW.items()
        }
        predicted_speaker = max(speaker_scores, key=speaker_scores.get)
        if max(speaker_scores.values()) < DETECTOR_THRESHOLD:
            predicted_speaker = "unknown"
        events.sort(key=lambda row: (row["start_seconds"], row["end_seconds"], row["label"]))
        rows.append(
            {
                "format": "qces_combined_zero_shot_detector_prediction_v1",
                "scene_id": str(scene["scene_id"]),
                "split": str(scene["split"]),
                "mixture_path": str(scene["mixture_path"]),
                "duration_seconds": duration,
                "predicted_events": events,
                "valid_frames": valid_frames,
                "frame_hop_seconds": FRAME_SECONDS,
                "target_frame_probabilities": {
                    **{
                        label: probabilities[:valid_frames, label_to_id[raw_label]].tolist()
                        for raw_label, label in RAW_TO_LABEL.items()
                    },
                    "Speech": speech_values[:valid_frames].tolist(),
                },
                "predicted_speaker": predicted_speaker,
                "speaker_scores": speaker_scores,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "detector",
                    "index": index,
                    "total": len(scenes),
                    "scene": scene["scene_id"],
                    "events": len(events),
                    "speaker": predicted_speaker,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    path = output / "detector_predictions.jsonl"
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt = {
        "format": "qces_combined_zero_shot_detector_receipt_v1",
        "complete": True,
        "model": "PretrainedSED BEATs-Strong public checkpoint",
        "checkpoint": portable(PRETRAINED_SED_ROOT / "resources/BEATs_strong_1.pt"),
        "checkpoint_sha256": sha256(PRETRAINED_SED_ROOT / "resources/BEATs_strong_1.pt"),
        "qces_finetuning": False,
        "fixed_decode_threshold": DETECTOR_THRESHOLD,
        "saved_target_frame_probabilities": True,
        "downstream_policy": "one global threshold may be selected on val only and then locked for test",
        "median_filter_frames": MEDIAN_FRAMES,
        "scene_count": len(rows),
        "predictions": portable(path),
    }
    atomic_text(output / "detector_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


def normalize_audio(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-6:
        return value
    return np.clip(value * min(10.0, (10.0 ** (-24.0 / 20.0)) / rms), -1.0, 1.0)


def asr_stage(
    dataset: Path,
    output: Path,
    model_id: str,
    *,
    predicted_spans: Path | None = None,
) -> None:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    scenes = read_jsonl(dataset / "scenes.jsonl")
    spans = (
        {str(row["scene_id"]): row for row in read_jsonl(predicted_spans)}
        if predicted_spans is not None
        else {}
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        local_files_only=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if torch.cuda.is_available() else -1,
    )
    rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        waveform, sample_rate = load_audio(resolve(str(scene["mixture_path"])))
        shift = 0.0
        input_scope = "complete_mixture_no_annotation_crop"
        if predicted_spans is not None:
            span = spans[str(scene["scene_id"])]["predicted_speech_span_seconds"]
            input_scope = "val_locked_predicted_speech_span_no_oracle_fallback"
            if span is None:
                rows.append(
                    {
                        "format": "qces_combined_predicted_span_phowhisper_prediction_v1",
                        "scene_id": str(scene["scene_id"]),
                        "split": str(scene["split"]),
                        "mixture_path": str(scene["mixture_path"]),
                        "hypothesis": "",
                        "words": [],
                        "input_scope": input_scope,
                        "predicted_speech_span_seconds": None,
                    }
                )
                print(json.dumps({"stage": "asr", "index": index, "total": len(scenes), "scene": scene["scene_id"], "words": 0, "text": "", "reason": "no_predicted_speech"}, ensure_ascii=False), flush=True)
                continue
            shift = max(0.0, float(span[0]))
            end_seconds = min(len(waveform) / sample_rate, float(span[1]))
            start_sample = int(math.floor(shift * sample_rate))
            end_sample = int(math.ceil(end_seconds * sample_rate))
            waveform = waveform[start_sample:end_sample]
        result = transcriber(
            {"array": normalize_audio(waveform), "sampling_rate": sample_rate},
            return_timestamps="word",
            generate_kwargs={
                "language": "vi",
                "task": "transcribe",
                "max_new_tokens": 128,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.05,
            },
        )
        words = []
        for chunk in result.get("chunks", []):
            timestamp = chunk.get("timestamp") or (None, None)
            if timestamp[0] is None:
                continue
            start = shift + max(0.0, float(timestamp[0]))
            end = shift + (float(timestamp[1]) if timestamp[1] is not None else float(timestamp[0]))
            words.append(
                {"text": str(chunk.get("text", "")).strip(), "start_seconds": start, "end_seconds": end}
            )
        rows.append(
            {
                "format": "qces_combined_phowhisper_prediction_v2",
                "scene_id": str(scene["scene_id"]),
                "split": str(scene["split"]),
                "mixture_path": str(scene["mixture_path"]),
                "hypothesis": str(result.get("text", "")).strip(),
                "words": words,
                "input_scope": input_scope,
                "predicted_speech_span_seconds": (
                    spans[str(scene["scene_id"])]["predicted_speech_span_seconds"]
                    if predicted_spans is not None
                    else None
                ),
            }
        )
        print(
            json.dumps(
                {
                    "stage": "asr",
                    "index": index,
                    "total": len(scenes),
                    "scene": scene["scene_id"],
                    "words": len(words),
                    "text": rows[-1]["hypothesis"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    filename = (
        "predicted_span_asr_predictions.jsonl"
        if predicted_spans is not None
        else "full_mixture_asr_predictions.jsonl"
    )
    path = output / filename
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt = {
        "format": "qces_combined_asr_receipt_v2",
        "complete": True,
        "model": model_id,
        "input_scope": (
            "val_locked_predicted_speech_span_no_oracle_fallback"
            if predicted_spans is not None
            else "complete_mixture_no_annotation_crop"
        ),
        "predicted_span_manifest": portable(predicted_spans) if predicted_spans else None,
        "scene_count": len(rows),
        "predictions": portable(path),
    }
    receipt_name = "predicted_span_asr_receipt.json" if predicted_spans else "asr_receipt.json"
    atomic_text(output / receipt_name, json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)


def edit_distance(reference: str, hypothesis: str) -> tuple[int, int]:
    left, right = word_tokens(reference), word_tokens(hypothesis)
    previous = list(range(len(right) + 1))
    for row, source in enumerate(left, 1):
        current = [row]
        for column, target in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (source != target))
            )
        previous = current
    return previous[-1], max(1, len(left))


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def group_inventory(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        label = str(event["label"])
        item = grouped.setdefault(
            label,
            {
                "label": label,
                "display_label": str(event.get("display_label") or DISPLAY.get(label, label)),
                "score": float(event.get("confidence", 1.0)),
                "occurrences": [],
            },
        )
        item["score"] = max(item["score"], float(event.get("confidence", 1.0)))
        item["occurrences"].append(
            {
                "start_seconds": float(event["start_seconds"]),
                "end_seconds": float(event["end_seconds"]),
            }
        )
    return list(grouped.values())


def decode_saved_probabilities(
    detector_row: Mapping[str, Any], threshold: float
) -> list[dict[str, Any]]:
    """Decode annotation-free detector scores with one globally locked threshold."""

    valid_frames = int(detector_row["valid_frames"])
    duration = float(detector_row["duration_seconds"])
    output: list[dict[str, Any]] = []
    for label, raw_values in detector_row["target_frame_probabilities"].items():
        values = np.asarray(raw_values, dtype=np.float32)
        for start, end in contiguous_segments(
            values,
            threshold=threshold,
            valid_frames=valid_frames,
            merge_gap_seconds=0.30 if label == "Speech" else 0.16,
            minimum_seconds=0.30 if label == "Speech" else 0.08,
        ):
            output.append(
                {
                    "label": str(label),
                    "display_label": DISPLAY[str(label)],
                    "start_seconds": start * FRAME_SECONDS,
                    "end_seconds": min(duration, end * FRAME_SECONDS),
                    "confidence": float(values[start:end].max()),
                    "mean_confidence": float(values[start:end].mean()),
                    "source": "public_beats_strong_val_locked_threshold",
                }
            )
    output.sort(key=lambda row: (row["start_seconds"], row["end_seconds"], row["label"]))
    return output


def boundary_answer(intent: str, inventory: list[dict[str, Any]]) -> str:
    flat = []
    for item in inventory:
        for occurrence in item["occurrences"]:
            flat.append({**occurrence, "label": item["label"], "display_label": item["display_label"]})
    speech = sorted((row for row in flat if row["label"] == "Speech"), key=lambda row: row["start_seconds"])
    if not speech:
        return "NONE"
    anchor = speech[0]
    candidates = [row for row in flat if row["label"] != "Speech"]
    if intent == "before":
        candidates = [row for row in candidates if row["end_seconds"] <= anchor["start_seconds"] + 0.05]
        result = max(candidates, key=lambda row: row["end_seconds"], default=None)
    else:
        candidates = [row for row in candidates if row["start_seconds"] >= anchor["end_seconds"] - 0.05]
        result = min(candidates, key=lambda row: row["start_seconds"], default=None)
    return "NONE" if result is None else str(result["label"])


def event_metrics(predicted: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: list[tuple[float, int, int]] = []
    for pred_id, pred in enumerate(predicted):
        for gold_id, target in enumerate(gold):
            if pred["label"] != target["label"]:
                continue
            score = interval_iou(
                (float(pred["start_seconds"]), float(pred["end_seconds"])),
                (float(target["start_seconds"]), float(target["end_seconds"])),
            )
            if score >= 0.30:
                candidates.append((score, pred_id, gold_id))
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    scores: list[float] = []
    for score, pred_id, gold_id in sorted(candidates, reverse=True):
        if pred_id in used_pred or gold_id in used_gold:
            continue
        used_pred.add(pred_id)
        used_gold.add(gold_id)
        scores.append(score)
    tp, fp, fn = len(scores), len(predicted) - len(scores), len(gold) - len(scores)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "event_precision_↑": precision,
        "event_recall_↑": recall,
        "event_f1_↑": 2 * precision * recall / max(precision + recall, 1e-12),
        "matched_mean_iou_↑": float(np.mean(scores)) if scores else 0.0,
    }


def speech_span_stage(dataset: Path, output: Path) -> None:
    """Calibrate one VAD threshold on val and emit annotation-free test spans."""

    scenes = {str(row["scene_id"]): row for row in read_jsonl(dataset / "scenes.jsonl")}
    detector = {str(row["scene_id"]): row for row in read_jsonl(output / "detector_predictions.jsonl")}
    grid = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    table: list[dict[str, Any]] = []
    for threshold in grid:
        totals = Counter()
        iou_sum = 0.0
        for scene_id, scene in scenes.items():
            if scene["split"] != "val":
                continue
            predicted = [row for row in decode_saved_probabilities(detector[scene_id], threshold) if row["label"] == "Speech"]
            gold_event = next(event for event in scene["events"] if event["event_kind"] == "speech")
            gold = [{"label": "Speech", "start_seconds": float(gold_event["onset_seconds"]), "end_seconds": float(gold_event["offset_seconds"])}]
            metrics = event_metrics(predicted, gold)
            totals.update({key: int(metrics[key]) for key in ("tp", "fp", "fn")})
            iou_sum += float(metrics["matched_mean_iou_↑"]) * int(metrics["tp"])
        precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
        recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
        table.append({"threshold": threshold, "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"], "precision_↑": precision, "recall_↑": recall, "f1_↑": 2 * precision * recall / max(precision + recall, 1e-12), "matched_mean_iou_↑": iou_sum / max(totals["tp"], 1)})
    selected = max(table, key=lambda row: (row["f1_↑"], row["matched_mean_iou_↑"], row["precision_↑"], row["threshold"]))
    threshold = float(selected["threshold"])
    rows: list[dict[str, Any]] = []
    for scene_id, scene in scenes.items():
        candidates = [row for row in decode_saved_probabilities(detector[scene_id], threshold) if row["label"] == "Speech"]
        chosen = max(
            candidates,
            key=lambda row: (
                (float(row["end_seconds"]) - float(row["start_seconds"])) * float(row["mean_confidence"]),
                float(row["confidence"]),
            ),
            default=None,
        )
        span = None
        if chosen is not None:
            span = [max(0.0, float(chosen["start_seconds"]) - 0.15), min(float(scene["duration_seconds"]), float(chosen["end_seconds"]) + 0.15)]
        rows.append({"format": "qces_val_locked_speech_span_prediction_v1", "scene_id": scene_id, "split": scene["split"], "predicted_speech_span_seconds": span, "confidence": None if chosen is None else float(chosen["confidence"]), "threshold": threshold, "selection": "max_duration_times_mean_confidence_then_pad_150ms"})
    path = output / "speech_span_predictions.jsonl"
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt = {"format": "qces_val_locked_speech_span_receipt_v1", "complete": True, "model": "public BEATs-Strong Speech-family frame scores", "qces_finetuning": False, "threshold_selection": "validation only", "selected": selected, "grid": table, "test_annotations_used_for_selection": False, "oracle_fallback": False, "predictions": portable(path)}
    atomic_text(output / "speech_span_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def evaluate_stage(dataset: Path, output: Path) -> None:
    scenes = {str(row["scene_id"]): row for row in read_jsonl(dataset / "scenes.jsonl")}
    detector = {str(row["scene_id"]): row for row in read_jsonl(output / "detector_predictions.jsonl")}
    asr_path = output / "predicted_span_asr_predictions.jsonl"
    if not asr_path.exists():
        asr_path = output / "full_mixture_asr_predictions.jsonl"
    asr = {str(row["scene_id"]): row for row in read_jsonl(asr_path)}

    # A single detector threshold is selected using validation annotations
    # only, then frozen for the test split.  There is no per-class or per-scene
    # adjustment, and the test labels never participate in this decision.
    threshold_grid = (0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    calibration: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        totals = Counter()
        iou_sum = 0.0
        for scene_id, scene in scenes.items():
            if scene["split"] != "val":
                continue
            predicted = [
                row
                for row in decode_saved_probabilities(detector[scene_id], threshold)
                if row["label"] != "Speech"
            ]
            gold = [
                {
                    "label": str(event["label"]),
                    "start_seconds": float(event["onset_seconds"]),
                    "end_seconds": float(event["offset_seconds"]),
                }
                for event in scene["events"]
                if event["event_kind"] != "speech"
            ]
            metrics = event_metrics(predicted, gold)
            totals.update({key: int(metrics[key]) for key in ("tp", "fp", "fn")})
            iou_sum += float(metrics["matched_mean_iou_↑"]) * int(metrics["tp"])
        precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
        recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
        calibration.append(
            {
                "threshold": threshold,
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "precision_↑": precision,
                "recall_↑": recall,
                "f1_↑": 2 * precision * recall / max(precision + recall, 1e-12),
                "matched_mean_iou_↑": iou_sum / max(totals["tp"], 1),
            }
        )
    selected_calibration = max(
        calibration,
        key=lambda row: (row["f1_↑"], row["precision_↑"], row["threshold"]),
    )
    selected_threshold = float(selected_calibration["threshold"])
    combined_rows: list[dict[str, Any]] = []
    split_accumulator: dict[str, dict[str, Any]] = {
        split: {
            "event_tp": 0,
            "event_fp": 0,
            "event_fn": 0,
            "event_iou_sum": 0.0,
            "edits": 0,
            "words": 0,
            "span_iou": [],
            "qa": Counter(),
        }
        for split in ("val", "test")
    }
    for scene_id, scene in scenes.items():
        detector_row = detector[scene_id]
        asr_row = asr[scene_id]
        predicted_events = [
            dict(row)
            for row in decode_saved_probabilities(detector_row, selected_threshold)
            if row["label"] != "Speech"
        ]
        words = list(asr_row["words"])
        beats_speech = [row for row in detector_row["predicted_events"] if row["label"] == "Speech"]
        predicted_asr_span = asr_row.get("predicted_speech_span_seconds")
        if predicted_asr_span is not None:
            speech_start = float(predicted_asr_span[0])
            speech_end = float(predicted_asr_span[1])
            speech_source = "public_beats_val_locked_predicted_span"
            speech_confidence = 1.0
        elif words:
            speech_start = max(0.0, float(words[0]["start_seconds"]) - 0.15)
            speech_end = min(float(scene["duration_seconds"]), float(words[-1]["end_seconds"]) + 0.15)
            speech_source = "phowhisper_full_mixture_word_timestamps"
            speech_confidence = 1.0
        elif beats_speech:
            speech_start = min(float(row["start_seconds"]) for row in beats_speech)
            speech_end = max(float(row["end_seconds"]) for row in beats_speech)
            speech_source = "public_beats_strong_zero_shot_fallback"
            speech_confidence = max(float(row["confidence"]) for row in beats_speech)
        else:
            speech_start = speech_end = 0.0
            speech_source = "no_speech_prediction"
            speech_confidence = 0.0
        if speech_end > speech_start:
            predicted_events.append(
                {
                    "label": "Speech",
                    "display_label": DISPLAY["Speech"],
                    "start_seconds": speech_start,
                    "end_seconds": speech_end,
                    "confidence": speech_confidence,
                    "mean_confidence": speech_confidence,
                    "source": speech_source,
                    "speaker_group": detector_row["predicted_speaker"],
                }
            )
        predicted_events.sort(key=lambda row: (row["start_seconds"], row["end_seconds"], row["label"]))
        gold_events = [
            {
                "label": "Speech" if event["event_kind"] == "speech" else str(event["label"]),
                "display_label": DISPLAY.get(str(event["label"]), str(event["display_name"])),
                "start_seconds": float(event["onset_seconds"]),
                "end_seconds": float(event["offset_seconds"]),
            }
            for event in scene["events"]
        ]
        speech_gold = next(event for event in scene["events"] if event["event_kind"] == "speech")
        reference = str(speech_gold["transcript"])
        hypothesis = str(asr_row["hypothesis"])
        edits, reference_words = edit_distance(reference, hypothesis)
        speech_iou = interval_iou(
            (speech_start, speech_end),
            (float(speech_gold["onset_seconds"]), float(speech_gold["offset_seconds"])),
        )
        predicted_inventory = group_inventory(predicted_events)
        gold_inventory = group_inventory(gold_events)
        list_pred = answer_event_graph_program(ParsedAudioQuestion("list"), predicted_inventory).answer
        list_gold = answer_event_graph_program(ParsedAudioQuestion("list"), gold_inventory).answer
        qa = {
            "list": list_pred == list_gold,
            "before_speech": boundary_answer("before", predicted_inventory)
            == boundary_answer("before", gold_inventory),
            "after_speech": boundary_answer("after", predicted_inventory)
            == boundary_answer("after", gold_inventory),
        }
        split = str(scene["split"])
        accumulator = split_accumulator[split]
        scene_event_metrics = event_metrics(predicted_events, gold_events)
        accumulator["event_tp"] += int(scene_event_metrics["tp"])
        accumulator["event_fp"] += int(scene_event_metrics["fp"])
        accumulator["event_fn"] += int(scene_event_metrics["fn"])
        accumulator["event_iou_sum"] += (
            float(scene_event_metrics["matched_mean_iou_↑"])
            * int(scene_event_metrics["tp"])
        )
        accumulator["edits"] += edits
        accumulator["words"] += reference_words
        accumulator["span_iou"].append(speech_iou)
        accumulator["qa"].update({key: int(value) for key, value in qa.items()})
        accumulator["qa"]["scenes"] += 1
        combined_rows.append(
            {
                "format": "qces_combined_honest_prediction_v1",
                "scene_id": scene_id,
                "split": split,
                "mixture_path": str(scene["mixture_path"]),
                "duration_seconds": float(scene["duration_seconds"]),
                "predicted_events": predicted_events,
                "predicted_inventory": predicted_inventory,
                "predicted_speaker": detector_row["predicted_speaker"],
                "speaker_scores": detector_row["speaker_scores"],
                "speech_span_source": speech_source,
                "speech_span_seconds": [speech_start, speech_end] if speech_end > speech_start else None,
                "hypothesis": hypothesis,
                "words": words,
                "wer_↓": edits / reference_words,
                "speech_span_iou_↑": speech_iou,
                "qa_correct": qa,
            }
        )
    summary: dict[str, Any] = {}
    for split, accumulator in split_accumulator.items():
        tp = int(accumulator["event_tp"])
        fp = int(accumulator["event_fp"])
        fn = int(accumulator["event_fn"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "event_precision_↑": precision,
            "event_recall_↑": recall,
            "event_f1_↑": 2 * precision * recall / max(precision + recall, 1e-12),
            "matched_mean_iou_↑": accumulator["event_iou_sum"] / max(tp, 1),
        }
        scenes_count = int(accumulator["qa"]["scenes"])
        summary[split] = {
            **metrics,
            "corpus_wer_↓": accumulator["edits"] / max(accumulator["words"], 1),
            "speech_span_mean_iou_↑": float(np.mean(accumulator["span_iou"])),
            "list_exact_accuracy_↑": accumulator["qa"]["list"] / max(scenes_count, 1),
            "before_speech_accuracy_↑": accumulator["qa"]["before_speech"] / max(scenes_count, 1),
            "after_speech_accuracy_↑": accumulator["qa"]["after_speech"] / max(scenes_count, 1),
            "scenes": scenes_count,
        }
    prediction_path = output / "combined_predictions.jsonl"
    atomic_text(
        prediction_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in combined_rows),
    )
    eval_source_ids = {
        split: {
            str(event["source_id"])
            for scene in scenes.values()
            if scene["split"] == split
            for event in scene["events"]
        }
        for split in ("val", "test")
    }
    rejected_checkpoint_train = (
        PROJECT_ROOT
        / "outputs/qces_full188_overlap_gold_natural_v3/detector_scene_manifest_single_train_gold.jsonl"
    )
    rejected_train_ids: set[str] = set()
    if rejected_checkpoint_train.exists():
        rejected_train_ids = {
            str(event["source_id"])
            for row in read_jsonl(rejected_checkpoint_train)
            for event in row.get("events", [])
            if event.get("source_id")
        }
    receipt = {
        "format": "qces_combined_honest_zero_shot_receipt_v1",
        "complete": True,
        "inference_contract": {
            "event_detector": (
                "public BEATs-Strong, one global threshold selected on val and locked for test, "
                "no QCES fine-tuning"
            ),
            "speech_text_and_span": (
                "public BEATs val-locked predicted speech span, then PhoWhisper-large on that crop; "
                "no oracle crop"
            ),
            "annotations_available_to_inference": False,
            "test_specific_parameter_tuning": False,
            "oracle_fallback": False,
        },
        "integrity": {
            "dataset_cross_split_source_leakage": len(
                eval_source_ids["val"] & eval_source_ids["test"]
            ),
            "qces_checkpoint_used": False,
            "rejected_qces_188_checkpoint_test_source_overlap": len(
                eval_source_ids["test"] & rejected_train_ids
            ),
            "rejected_qces_188_checkpoint_reason": (
                "Its training manifest contains evaluation source IDs, so it is intentionally "
                "not used by this demo."
            ),
            "important_limitation": (
                "Environmental sources are AudioSet-derived; overlap with the public BEATs pretraining "
                "corpus cannot be excluded. Results are zero-shot for this QCES task but not claimed as "
                "strict pretraining-source-disjoint generalization."
            ),
        },
        "detector_threshold_selection": {
            "protocol": "maximize environmental event F1 on val only; lock before test evaluation",
            "global_not_per_class": True,
            "selected_threshold": selected_threshold,
            "selected_val_row": selected_calibration,
            "grid": calibration,
            "fixed_public_threshold_0.5_was_not_erased": True,
        },
        "summary": summary,
        "predictions": portable(prediction_path),
        "detector_receipt": portable(output / "detector_receipt.json"),
        "speech_span_receipt": portable(output / "speech_span_receipt.json"),
        "asr_receipt": portable(
            output
            / (
                "predicted_span_asr_receipt.json"
                if (output / "predicted_span_asr_receipt.json").exists()
                else "asr_receipt.json"
            )
        ),
    }
    atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("detector", "speech-spans", "asr", "asr-predicted", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-model", default=MODEL_ID)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stages = (
        ("detector", "speech-spans", "asr-predicted", "evaluate")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        if stage == "detector":
            detector_stage(dataset, output, args.device)
        elif stage == "speech-spans":
            speech_span_stage(dataset, output)
        elif stage == "asr":
            asr_stage(dataset, output, args.asr_model)
        elif stage == "asr-predicted":
            asr_stage(
                dataset,
                output,
                args.asr_model,
                predicted_spans=output / "speech_span_predictions.jsonl",
            )
        else:
            evaluate_stage(dataset, output)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
