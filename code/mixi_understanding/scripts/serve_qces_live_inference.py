#!/usr/bin/env python3
"""Persistent annotation-free inference service for live automotive AudioQA."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from transformers import AutoFeatureExtractor, WavLMForXVector

from mixi_understanding.scripts.infer_qces_combined_honest_v1 import (
    DISPLAY,
    FRAME_SECONDS,
    GENDER_RAW,
    MEDIAN_FRAMES,
    PRETRAINED_SED_ROOT,
    RAW_TO_LABEL,
    SPEECH_RAW,
    contiguous_segments,
    load_audio,
    normalize_audio,
)


ENVIRONMENT_THRESHOLD = 0.05
SPEECH_THRESHOLD = 0.10
MAX_AUDIO_SECONDS = 10.0
DEFAULT_ASR_MODEL = (
    "/home/baoltq/.cache/huggingface/hub/models--vinai--PhoWhisper-tiny/"
    "snapshots/cc51d32be916efebde04ff549854fa1741cb5c02"
)
HUMAN_EVENT_THRESHOLDS = {
    # Selected only on AudioSet-Strong dev by event F1; test labels were not
    # used.  These conservative operating points avoid flooding the live
    # timeline with the low-threshold false positives seen in the audit.
    "Laughter": 0.20,
    "Giggle": 0.05,
    "Conversation": 0.30,
    "Shout": 0.15,
    "Crying_and_sobbing": 0.15,
    # First gradual ontology expansion. Keep animal thresholds conservative
    # until they are calibrated on a dedicated, source-disjoint live set.
    "Bark": 0.20,
    "Howl": 0.20,
    "Whimper_(dog)": 0.20,
    "Meow": 0.20,
    "Purr": 0.20,
    "Caterwaul": 0.20,
    # Engine-family expansion.  Both operating points are conservative with
    # respect to the two live recordings that exposed the missing subtype:
    # their peaks are 0.510/0.171 and 0.022/0.159 respectively.
    "Accelerating_and_revving": 0.10,
    "Medium_engine_(mid_frequency)": 0.10,
}
SPEAKER_MODEL_ID = "microsoft/wavlm-base-plus-sv"
DIARIZATION_WINDOW_SECONDS = 1.50
DIARIZATION_HOP_SECONDS = 0.25
DIARIZATION_MIN_SILHOUETTE = 0.25
DIARIZATION_MIN_WINDOWS_PER_SPEAKER = 3
SPEAKER_VERIFIER_URL = "http://127.0.0.1:8517/verify"
SPEAKER_VERIFIER_TIMEOUT_SECONDS = 120


class SpeakerDiarizer:
    """Small annotation-free two-speaker diarizer for the live demo.

    WavLM speaker-verification embeddings are clustered over overlapping
    speech windows.  A second speaker is emitted only when it forms a stable
    run and the cosine silhouette clears a fixed, target-free threshold.
    """

    def __init__(self, device_name: str = "cpu") -> None:
        self.device = torch.device(device_name)
        self.available = True
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(
                SPEAKER_MODEL_ID, local_files_only=True,
            )
            self.model = WavLMForXVector.from_pretrained(
                SPEAKER_MODEL_ID, local_files_only=True,
            ).to(self.device).eval()
            print(f"[ready] speaker diarization on {self.device}", flush=True)
        except (OSError, RuntimeError) as error:
            # Keep event detection and Vietnamese ASR available when the
            # optional WavLM cache is unavailable after a reboot.
            self.available = False
            self.extractor = None
            self.model = None
            print(
                f"[warning] WavLM speaker diarization unavailable: {error}",
                flush=True,
            )

    @torch.inference_mode()
    def _embeddings(self, windows: list[np.ndarray]) -> np.ndarray:
        values: list[np.ndarray] = []
        for start in range(0, len(windows), 8):
            batch = self.extractor(
                windows[start : start + 8],
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            inputs = {
                key: value.to(self.device)
                for key, value in batch.items()
                if key in {"input_values", "attention_mask"}
            }
            embedding = self.model(**inputs).embeddings
            embedding = F.normalize(embedding.float(), dim=-1)
            values.append(embedding.cpu().numpy())
        return np.concatenate(values, axis=0)

    def diarize(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        speech_segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not speech_segments:
            return {
                "speaker_count": 0,
                "turns": [],
                "silhouette": None,
                "multi_speaker_accepted": False,
            }
        if not self.available:
            return {
                "speaker_count": 1,
                "turns": [dict(segment) | {"speaker_id": "SPEAKER_01"} for segment in speech_segments],
                "silhouette": None,
                "multi_speaker_accepted": False,
                "warning": "WavLM speaker model chưa sẵn sàng; tạm thời gộp lời nói thành một cụm.",
            }
        if sample_rate != 16_000:
            raise ValueError("speaker diarizer expects 16 kHz audio")

        window_samples = int(round(DIARIZATION_WINDOW_SECONDS * sample_rate))
        hop = DIARIZATION_HOP_SECONDS
        windows: list[np.ndarray] = []
        window_rows: list[dict[str, Any]] = []
        for segment_index, segment in enumerate(speech_segments):
            segment_start = float(segment["start_seconds"])
            segment_end = float(segment["end_seconds"])
            latest = max(segment_start, segment_end - DIARIZATION_WINDOW_SECONDS)
            starts = list(np.arange(segment_start, latest + 1e-7, hop))
            if not starts or latest - starts[-1] > 0.08:
                starts.append(latest)
            for start_seconds in starts:
                left = int(round(start_seconds * sample_rate))
                right = min(len(waveform), left + window_samples)
                value = np.asarray(waveform[left:right], dtype=np.float32)
                if len(value) < int(0.50 * sample_rate):
                    continue
                if len(value) < window_samples:
                    value = np.pad(value, (0, window_samples - len(value)))
                windows.append(value)
                window_rows.append(
                    {
                        "segment_index": segment_index,
                        "start_seconds": start_seconds,
                        "center_seconds": min(
                            segment_end,
                            start_seconds + DIARIZATION_WINDOW_SECONDS / 2.0,
                        ),
                    }
                )

        if not windows:
            turns = [dict(segment) | {"speaker_id": "SPEAKER_01"} for segment in speech_segments]
            return {
                "speaker_count": 1,
                "turns": turns,
                "silhouette": None,
                "multi_speaker_accepted": False,
            }

        embeddings = self._embeddings(windows)
        cluster = np.zeros(len(windows), dtype=np.int64)
        score: float | None = None
        accepted = False
        if len(windows) >= 2 * DIARIZATION_MIN_WINDOWS_PER_SPEAKER:
            candidate = AgglomerativeClustering(
                n_clusters=2,
                metric="cosine",
                linkage="average",
            ).fit_predict(embeddings)
            counts = np.bincount(candidate, minlength=2)
            score = float(silhouette_score(embeddings, candidate, metric="cosine"))
            accepted = (
                int(counts.min()) >= DIARIZATION_MIN_WINDOWS_PER_SPEAKER
                and score >= DIARIZATION_MIN_SILHOUETTE
            )
            if accepted:
                cluster = candidate.astype(np.int64)

        # Remove one-window cluster flicker without erasing a stable short turn.
        for segment_index in range(len(speech_segments)):
            indices = [
                index for index, row in enumerate(window_rows)
                if int(row["segment_index"]) == segment_index
            ]
            if len(indices) >= 3:
                cluster[indices] = median_filter(cluster[indices], size=3, mode="nearest")

        first_seen: dict[int, float] = {}
        for label, row in zip(cluster.tolist(), window_rows):
            first_seen.setdefault(int(label), float(row["center_seconds"]))
        ordered_clusters = sorted(first_seen, key=first_seen.get)
        speaker_name = {
            label: f"SPEAKER_{index + 1:02d}"
            for index, label in enumerate(ordered_clusters)
        }

        raw_turns: list[dict[str, Any]] = []
        for segment_index, parent in enumerate(speech_segments):
            rows = [
                (row, int(cluster[index]))
                for index, row in enumerate(window_rows)
                if int(row["segment_index"]) == segment_index
            ]
            rows.sort(key=lambda item: float(item[0]["center_seconds"]))
            if not rows:
                raw_turns.append(dict(parent) | {"speaker_id": "SPEAKER_01"})
                continue
            centers = [float(row["center_seconds"]) for row, _ in rows]
            boundaries = [float(parent["start_seconds"])]
            boundaries.extend(
                (centers[index] + centers[index + 1]) / 2.0
                for index in range(len(centers) - 1)
            )
            boundaries.append(float(parent["end_seconds"]))
            for index, (_, label) in enumerate(rows):
                start_seconds = boundaries[index]
                end_seconds = boundaries[index + 1]
                if end_seconds - start_seconds < 0.08:
                    continue
                raw_turns.append(
                    dict(parent)
                    | {
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "speaker_id": speaker_name[label],
                    }
                )

        turns: list[dict[str, Any]] = []
        for turn in raw_turns:
            if (
                turns
                and turns[-1]["speaker_id"] == turn["speaker_id"]
                and float(turn["start_seconds"]) - float(turns[-1]["end_seconds"]) <= 0.05
            ):
                turns[-1]["end_seconds"] = float(turn["end_seconds"])
            else:
                turns.append(dict(turn))
        active_speakers = sorted({str(turn["speaker_id"]) for turn in turns})
        return {
            "speaker_count": len(active_speakers),
            "turns": turns,
            "silhouette": score,
            "multi_speaker_accepted": accepted,
            "window_seconds": DIARIZATION_WINDOW_SECONDS,
            "hop_seconds": DIARIZATION_HOP_SECONDS,
            "minimum_silhouette": DIARIZATION_MIN_SILHOUETTE,
        }


class LiveBackend:
    def __init__(
        self,
        device_name: str,
        asr_device_name: str,
        asr_model_path: str,
        asr_backend: str,
    ) -> None:
        self.asr_backend = asr_backend
        if asr_backend == "chunkformer":
            from chunkformer import ChunkFormerModel
        else:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import models.prediction_wrapper as prediction_wrapper
        from data_util.audioset_classes import as_strong_train_classes
        from models.beats.BEATs_wrapper import BEATsWrapper
        from models.prediction_wrapper import PredictionsWrapper

        prediction_wrapper.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.asr_device = torch.device(
            asr_device_name if torch.cuda.is_available() or asr_device_name == "cpu" else "cpu"
        )
        self.detector = PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
        self.detector.eval().to(self.device)
        self.label_to_id = {
            label: index for index, label in enumerate(as_strong_train_classes)
        }
        required = set(RAW_TO_LABEL) | set(SPEECH_RAW)
        missing = sorted(required - set(self.label_to_id))
        if missing:
            raise RuntimeError(f"public detector is missing labels: {missing}")

        if asr_backend == "chunkformer":
            self.transcriber = ChunkFormerModel.from_pretrained(
                asr_model_path,
                local_files_only=True,
            ).to(self.asr_device).eval()
        else:
            self.asr_processor = AutoProcessor.from_pretrained(
                asr_model_path,
                local_files_only=True,
            )
            self.transcriber = AutoModelForSpeechSeq2Seq.from_pretrained(
                asr_model_path,
                local_files_only=True,
                torch_dtype=torch.float32,
            ).to(self.asr_device).eval()
        self.diarizer = SpeakerDiarizer("cpu")
        self.lock = threading.Lock()
        print(
            f"[ready] event detector on {self.device}; {asr_backend} ASR on {self.asr_device}",
            flush=True,
        )

    @staticmethod
    def _verify_two_speakers(
        audio_path: Path,
        speech_segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "audio_path": str(audio_path.resolve()),
                "speech_segments": speech_segments,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            SPEAKER_VERIFIER_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=SPEAKER_VERIFIER_TIMEOUT_SECONDS,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            # A second speaker is a positive claim.  If the verifier is down,
            # prefer one speaker over repeating the previous false-positive
            # behaviour.
            return {
                "two_speakers_accepted": False,
                "decision": "one_speaker",
                "verifier_available": False,
                "error": f"{type(error).__name__}: {error}",
            }

    @staticmethod
    def _collapse_to_one_speaker(
        diarization: dict[str, Any],
        speech_segments: list[dict[str, Any]],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        collapsed = dict(diarization)
        collapsed.update(
            {
                "speaker_count": 1 if speech_segments else 0,
                "turns": [
                    dict(segment) | {"speaker_id": "SPEAKER_01"}
                    for segment in speech_segments
                ],
                "multi_speaker_accepted": False,
                "embedding_candidate_speaker_count": int(
                    diarization.get("speaker_count", 1)
                ),
                "speaker_verification": verification,
            }
        )
        return collapsed

    def _detect(self, waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
        original_duration = len(waveform) / sample_rate
        duration = min(original_duration, MAX_AUDIO_SECONDS)
        usable = waveform[: int(round(duration * sample_rate))]
        audio = torch.from_numpy(usable).unsqueeze(0)
        target_samples = int(MAX_AUDIO_SECONDS * sample_rate)
        if audio.shape[1] < target_samples:
            audio = F.pad(audio, (0, target_samples - audio.shape[1]))
        with torch.inference_mode():
            mel = self.detector.mel_forward(audio.to(self.device))
            logits, _ = self.detector(mel)
            probabilities = logits.sigmoid()[0].transpose(0, 1).float().cpu().numpy()
        probabilities = median_filter(
            probabilities, size=(MEDIAN_FRAMES, 1), mode="nearest"
        )
        valid_frames = min(
            probabilities.shape[0], int(math.ceil(duration / FRAME_SECONDS))
        )
        events: list[dict[str, Any]] = []
        class_scores: dict[str, dict[str, float]] = {}
        for raw_label, label in RAW_TO_LABEL.items():
            values = probabilities[:, self.label_to_id[raw_label]]
            valid_values = values[:valid_frames]
            class_scores[label] = {
                "peak": float(valid_values.max()) if len(valid_values) else 0.0,
                "mean": float(valid_values.mean()) if len(valid_values) else 0.0,
            }
            threshold = HUMAN_EVENT_THRESHOLDS.get(label, ENVIRONMENT_THRESHOLD)
            for start, end in contiguous_segments(
                values,
                threshold=threshold,
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
                        "threshold": threshold,
                        "source": "public_beats_strong_val_locked_threshold",
                    }
                )

        speech_values = np.max(
            np.stack(
                [probabilities[:, self.label_to_id[label]] for label in SPEECH_RAW],
                axis=1,
            ),
            axis=1,
        )
        speech_candidates: list[dict[str, Any]] = []
        for start, end in contiguous_segments(
            speech_values,
            threshold=SPEECH_THRESHOLD,
            valid_frames=valid_frames,
            merge_gap_seconds=0.30,
            minimum_seconds=0.30,
        ):
            speech_candidates.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "start_seconds": start * FRAME_SECONDS,
                    "end_seconds": min(duration, end * FRAME_SECONDS),
                    "confidence": float(speech_values[start:end].max()),
                    "mean_confidence": float(speech_values[start:end].mean()),
                }
            )

        speech_segments: list[dict[str, Any]] = []
        for candidate in speech_candidates:
            start_frame = int(candidate["start_frame"])
            end_frame = int(candidate["end_frame"])
            local_speaker_scores = {
                speaker: float(
                    probabilities[
                        start_frame:end_frame,
                        self.label_to_id[raw_label],
                    ].mean()
                )
                for speaker, raw_label in GENDER_RAW.items()
            }
            local_speaker = max(local_speaker_scores, key=local_speaker_scores.get)
            if max(local_speaker_scores.values(), default=0.0) < SPEECH_THRESHOLD:
                local_speaker = "unknown"
            segment = {
                "start_seconds": max(0.0, float(candidate["start_seconds"]) - 0.15),
                "end_seconds": min(duration, float(candidate["end_seconds"]) + 0.15),
                "confidence": float(candidate["confidence"]),
                "mean_confidence": float(candidate["mean_confidence"]),
                "predicted_speaker": local_speaker,
                "speaker_scores": local_speaker_scores,
            }
            speech_segments.append(segment)
            events.append(
                {
                    "label": "Speech",
                    "display_label": DISPLAY["Speech"],
                    "start_seconds": segment["start_seconds"],
                    "end_seconds": segment["end_seconds"],
                    "confidence": segment["confidence"],
                    "mean_confidence": segment["mean_confidence"],
                    "predicted_speaker": local_speaker,
                    "source": "public_beats_strong_val_locked_speech_threshold",
                }
            )

        speech_span = None
        if speech_segments:
            speech_span = [
                min(float(row["start_seconds"]) for row in speech_segments),
                max(float(row["end_seconds"]) for row in speech_segments),
            ]

        active = np.flatnonzero(speech_values[:valid_frames] >= SPEECH_THRESHOLD)
        speaker_scores = {
            speaker: float(
                probabilities[
                    active if len(active) else slice(0, valid_frames),
                    self.label_to_id[raw_label],
                ].mean()
            )
            for speaker, raw_label in GENDER_RAW.items()
        }
        known_speakers = {
            str(row["predicted_speaker"])
            for row in speech_segments
            if row["predicted_speaker"] != "unknown"
        }
        if len(known_speakers) > 1:
            predicted_speaker = "multiple"
        elif known_speakers:
            predicted_speaker = next(iter(known_speakers))
        else:
            predicted_speaker = "unknown"
        events.sort(key=lambda row: (row["start_seconds"], row["end_seconds"], row["label"]))
        return {
            "events": events,
            "class_scores": class_scores,
            "speech_span": speech_span,
            "speech_segments": speech_segments,
            "predicted_speaker": predicted_speaker,
            "speaker_scores": speaker_scores,
            "duration_seconds": duration,
            "original_duration_seconds": original_duration,
            "truncated_to_10_seconds": original_duration > MAX_AUDIO_SECONDS,
        }

    @staticmethod
    def _timestamp_seconds(value: str) -> float:
        parts = str(value).split(":")
        if len(parts) != 4:
            return 0.0
        hours, minutes, seconds, milliseconds = (int(part) for part in parts)
        return hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0

    def _transcribe(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        speech_span: list[float] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        if speech_span is None:
            return "", []
        shift, end_seconds = float(speech_span[0]), float(speech_span[1])
        crop = waveform[
            int(math.floor(shift * sample_rate)) : int(math.ceil(end_seconds * sample_rate))
        ]
        if not len(crop):
            return "", []
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            sf.write(
                temporary_path,
                normalize_audio(crop),
                sample_rate,
                subtype="PCM_16",
            )
            if self.asr_backend == "chunkformer":
                decoded = self.transcriber.endless_decode(
                    audio_path=str(temporary_path),
                    chunk_size=64,
                    left_context_size=128,
                    right_context_size=128,
                    total_batch_duration=60,
                    return_timestamps=True,
                )
            else:
                audio, rate = sf.read(temporary_path, dtype="float32", always_2d=False)
                prepared = self.asr_processor(audio, sampling_rate=rate, return_tensors="pt")
                inputs = {
                    key: value.to(self.asr_device)
                    for key, value in prepared.items()
                    if hasattr(value, "to")
                }
                with torch.inference_mode():
                    output = self.transcriber.generate(
                        **inputs,
                        language="vi",
                        task="transcribe",
                        max_new_tokens=256,
                    )
                decoded = self.asr_processor.batch_decode(
                    output, skip_special_tokens=True,
                )[0].strip()
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        timestamped = decoded if isinstance(decoded, list) else []
        phrases: list[str] = []
        words: list[dict[str, Any]] = []
        if isinstance(decoded, str) and decoded:
            timestamped = [{"decode": decoded, "start": "00:00:00:000", "end": "00:00:00:000"}]
        for item in timestamped:
            if not isinstance(item, dict):
                continue
            phrase = str(item.get("decode") or "").strip()
            if not phrase:
                continue
            phrases.append(phrase)
            local_start = self._timestamp_seconds(str(item.get("start") or ""))
            local_end = self._timestamp_seconds(str(item.get("end") or ""))
            if local_end <= local_start:
                local_start, local_end = 0.0, end_seconds - shift
            tokens = phrase.split()
            for index, token in enumerate(tokens):
                token_start = local_start + (local_end - local_start) * index / len(tokens)
                token_end = local_start + (local_end - local_start) * (index + 1) / len(tokens)
                words.append({
                    "text": token,
                    "start_seconds": shift + token_start,
                    "end_seconds": min(end_seconds, shift + token_end),
                })
        hypothesis = " ".join(phrases).strip()
        if not hypothesis and isinstance(decoded, str):
            hypothesis = decoded.strip()
        return hypothesis, words

    def infer(
        self,
        audio_path: Path,
        verify_speakers: bool = True,
    ) -> dict[str, Any]:
        waveform, sample_rate = load_audio(audio_path)
        if len(waveform) < int(0.25 * sample_rate):
            raise ValueError("audio must be at least 0.25 seconds")
        with self.lock:
            detection = self._detect(waveform, sample_rate)
            diarization = self.diarizer.diarize(
                waveform,
                sample_rate,
                list(detection["speech_segments"]),
            )
            if verify_speakers and int(diarization.get("speaker_count", 0)) >= 2:
                verification = self._verify_two_speakers(
                    audio_path,
                    list(detection["speech_segments"]),
                )
                if bool(verification.get("two_speakers_accepted")):
                    diarization["speaker_verification"] = verification
                    diarization["embedding_candidate_speaker_count"] = int(
                        diarization["speaker_count"]
                    )
                else:
                    diarization = self._collapse_to_one_speaker(
                        diarization,
                        list(detection["speech_segments"]),
                        verification,
                    )
            elif not verify_speakers:
                diarization["speaker_verification"] = {
                    "skipped": True,
                    "reason": "internal separated-stem transcription",
                }
            speech_segments: list[dict[str, Any]] = []
            all_words: list[dict[str, Any]] = []
            hypotheses: list[str] = []
            for segment in diarization["turns"]:
                hypothesis, words = self._transcribe(
                    waveform,
                    sample_rate,
                    [float(segment["start_seconds"]), float(segment["end_seconds"])],
                )
                enriched = dict(segment)
                enriched["hypothesis"] = hypothesis
                enriched["words"] = words
                speech_segments.append(enriched)
                if hypothesis:
                    hypotheses.append(hypothesis)
                all_words.extend(words)
            all_words.sort(key=lambda row: (row["start_seconds"], row["end_seconds"]))
            detection["events"] = [
                event for event in detection["events"] if event["label"] != "Speech"
            ]
            for segment in speech_segments:
                speaker_id = str(segment.get("speaker_id") or "SPEAKER_01")
                speaker_number = speaker_id.rsplit("_", 1)[-1].lstrip("0") or "1"
                detection["events"].append(
                    {
                        "label": "Speech",
                        "display_label": f"{DISPLAY['Speech']} · Người nói {speaker_number}",
                        "start_seconds": float(segment["start_seconds"]),
                        "end_seconds": float(segment["end_seconds"]),
                        "confidence": float(segment.get("confidence", 0.0)),
                        "mean_confidence": float(segment.get("mean_confidence", 0.0)),
                        "predicted_speaker": str(segment.get("predicted_speaker") or "unknown"),
                        "speaker_id": speaker_id,
                        "source": "wavlm_speaker_diarization_v1",
                    }
                )
            detection["events"].sort(
                key=lambda row: (row["start_seconds"], row["end_seconds"], row["label"])
            )
        return {
            "format": "qces_live_annotation_free_prediction_v3",
            "complete": True,
            "input_path": str(audio_path.resolve()),
            "models": {
                "event_and_speech_span": "sound-event detector",
                "speech_transcript": f"{self.asr_backend} Vietnamese ASR",
            },
            "thresholds": {
                "environment": ENVIRONMENT_THRESHOLD,
                "human_events": HUMAN_EVENT_THRESHOLDS,
                "speech": SPEECH_THRESHOLD,
                "selection": "locked from automotive validation; no live-audio tuning",
            },
            "annotations_used": False,
            "predicted_events": detection["events"],
            "class_scores": detection["class_scores"],
            "speech_span_seconds": detection["speech_span"],
            "speech_segments": speech_segments,
            "speaker_count": int(diarization["speaker_count"]),
            "speaker_turns": speech_segments,
            "diarization": {
                key: value for key, value in diarization.items() if key != "turns"
            },
            "predicted_speaker": detection["predicted_speaker"],
            "speaker_scores": detection["speaker_scores"],
            "hypothesis": " ".join(hypotheses).strip(),
            "words": all_words,
            "duration_seconds": detection["duration_seconds"],
            "original_duration_seconds": detection["original_duration_seconds"],
            "truncated_to_10_seconds": detection["truncated_to_10_seconds"],
        }


BACKEND: LiveBackend


class Handler(BaseHTTPRequestHandler):
    server_version = "QCESLiveInference/1.0"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(
                200,
                {
                    "ok": True,
                    "services": [
                        "sound-event detection",
                        "Vietnamese speech recognition",
                        "speaker diarization",
                    ],
                    "event_class_count": len(RAW_TO_LABEL) + 1,
                    "event_classes": [*RAW_TO_LABEL.values(), "Speech"],
                    "human_event_thresholds": HUMAN_EVENT_THRESHOLDS,
                    "max_audio_seconds": MAX_AUDIO_SECONDS,
                },
            )
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            audio_path = Path(str(payload.get("audio_path") or "")).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            output = BACKEND.infer(
                audio_path,
                verify_speakers=bool(payload.get("verify_speakers", True)),
            )
            self._send(200, output)
        except Exception as error:  # noqa: BLE001
            self._send(
                500,
                {"error": type(error).__name__, "message": str(error)},
            )

    def log_message(self, message: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {message % args}", flush=True)


def main() -> int:
    global BACKEND
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--asr-backend", choices=("chunkformer", "whisper"), default="chunkformer")
    args = parser.parse_args()
    BACKEND = LiveBackend(args.device, args.asr_device, args.asr_model, args.asr_backend)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
