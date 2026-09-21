#!/usr/bin/env python3
"""Serve annotation-free multi-speaker region clustering for the live demo.

The service deliberately keeps speaker *count* as an estimate.  It clusters
speaker-verification embeddings over time, masks windows dominated by detected
environmental events, and invokes the existing two-source separator only near
speaker transitions.  It never forces the number of speakers from metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from mixi_understanding.scripts.infer_qces_multispeaker_region_separator import (
    LIVE_URL,
    SAMPLE_RATE,
    _decode_turns,
    _event_spans,
    _fade_mask,
    _fit_length,
    _local_separation,
    _overlap,
    _post_json,
    _read_mono,
    _select_speaker_count,
    _speaker_region,
    _window_audio,
    _write,
)
from mixi_understanding.scripts.serve_qces_live_inference import SpeakerDiarizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "qces_multispeaker_region_live_v3"
SERVICE_VERSION = "qces-multispeaker-region-live-v3"
MAX_AUDIO_SECONDS = 10.1


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    value.update(SERVICE_VERSION.encode("utf-8"))
    return value.hexdigest()[:20]


def _valid_cached_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    speakers = list(receipt.get("speakers", []))
    if speakers and all(Path(str(row.get("audio_path") or "")).is_file() for row in speakers):
        return receipt
    return None


class MultiSpeakerRegionBackend:
    """Persistent WavLM clustering backend with cached waveform outputs."""

    def __init__(self) -> None:
        self.diarizer = SpeakerDiarizer("cpu")
        self.lock = threading.Lock()

    def separate(
        self,
        audio_path: Path,
        prediction: dict[str, Any] | None = None,
        max_speakers: int = 4,
    ) -> dict[str, Any]:
        run_dir = OUTPUT_ROOT / _digest(audio_path)
        receipt_path = run_dir / "receipt.json"
        cached = _valid_cached_receipt(receipt_path)
        if cached is not None:
            return cached

        with self.lock:
            cached = _valid_cached_receipt(receipt_path)
            if cached is not None:
                return cached

            waveform = _read_mono(audio_path)
            duration = len(waveform) / SAMPLE_RATE
            if duration < 0.5:
                raise ValueError("audio must be at least 0.5 seconds")
            if duration > MAX_AUDIO_SECONDS:
                raise ValueError(
                    f"multi-speaker live separation supports at most {MAX_AUDIO_SECONDS:.1f} seconds"
                )
            if prediction is None:
                prediction = _post_json(
                    LIVE_URL,
                    {"audio_path": str(audio_path.resolve()), "verify_speakers": False},
                    timeout=180,
                )
            if not isinstance(prediction, dict):
                raise TypeError("prediction must be an object")

            speech_start, speech_end = _speaker_region(prediction, duration)
            if speech_end - speech_start < 0.5:
                raise ValueError("no sufficiently long speech region was detected")
            event_spans = _event_spans(prediction)
            window_seconds = 1.0
            hop_seconds = 0.20
            latest = max(speech_start, speech_end - window_seconds)
            starts = list(np.arange(speech_start, latest + 1e-7, hop_seconds))
            if not starts or latest - starts[-1] > 0.04:
                starts.append(latest)
            starts = sorted(set(round(float(value), 6) for value in starts))
            windows = _window_audio(waveform, starts, window_seconds)
            embeddings = self.diarizer._embeddings(windows)
            keep = np.asarray(
                [
                    max(
                        [
                            _overlap(
                                start,
                                start + window_seconds,
                                event_start,
                                event_end,
                            )
                            for event_start, event_end, _ in event_spans
                        ]
                        or [0.0]
                    )
                    / window_seconds
                    < 0.25
                    for start in starts
                ],
                dtype=bool,
            )
            minimum_unmasked_windows = max(4, int(np.ceil(0.35 * len(keep))))
            event_masking_fallback = int(keep.sum()) < minimum_unmasked_windows
            if event_masking_fallback:
                # A false environmental event can overlap almost all speech
                # (e.g. the live detector labelled speech as long Clapping).
                # In that case masking destroys short-speaker recall, so use
                # all speech windows and keep the event spans only for audit.
                keep[:] = True

            speaker_count, eligible_labels, count_audit = _select_speaker_count(
                embeddings,
                keep,
                max(1, min(int(max_speakers), 4)),
                minimum_cluster_windows=2,
                # Web uploads are commonly decoded from MP3/M4A.  A broad
                # 0.02 parsimony band changed the same perceptual scene from
                # four clusters (WAV) to two (MP3) despite k=4 having the best
                # silhouette.  Keep only near-exact ties conservative.
                complexity_tolerance=0.003,
            )
            turns, centroids, _ = _decode_turns(
                embeddings,
                starts,
                keep,
                eligible_labels,
                speech_start,
                speech_end,
                window_seconds,
            )
            speaker_count = len({int(row["speaker_index"]) for row in turns})
            if speaker_count < 1:
                raise RuntimeError("speaker clustering returned no active speaker")

            run_dir.mkdir(parents=True, exist_ok=True)
            stems = [np.zeros_like(waveform) for _ in range(speaker_count)]
            for turn in turns:
                left = int(round(float(turn["start_seconds"]) * SAMPLE_RATE))
                right = int(round(float(turn["end_seconds"]) * SAMPLE_RATE))
                stems[int(turn["speaker_index"])] += waveform * _fade_mask(
                    left, right, len(waveform)
                )

            transition_audits: list[dict[str, Any]] = []
            for transition_index, (previous, current) in enumerate(
                zip(turns, turns[1:]), start=1
            ):
                old_speaker = int(previous["speaker_index"])
                new_speaker = int(current["speaker_index"])
                if old_speaker == new_speaker:
                    continue
                boundary = (
                    float(previous["end_seconds"]) + float(current["start_seconds"])
                ) / 2.0
                start = max(speech_start, boundary - 0.65)
                end = min(speech_end, boundary + 0.65)
                try:
                    assignment, audit, should_apply = _local_separation(
                        audio_path,
                        run_dir,
                        transition_index,
                        waveform,
                        waveform,
                        start,
                        end,
                        (old_speaker, new_speaker),
                        centroids,
                        self.diarizer,
                        0.75,
                        0.40,
                    )
                    if should_apply:
                        left = int(round(start * SAMPLE_RATE))
                        right = min(
                            len(waveform), left + len(next(iter(assignment.values())))
                        )
                        blend = np.sin(
                            np.linspace(0.0, np.pi, right - left, dtype=np.float32)
                        ) ** 2
                        for speaker in (old_speaker, new_speaker):
                            local = assignment[speaker][: right - left]
                            stems[speaker][left:right] = (
                                stems[speaker][left:right] * (1.0 - blend)
                                + local * blend
                            )
                except (OSError, RuntimeError, urllib.error.URLError) as error:
                    audit = {
                        "transition_index": transition_index,
                        "start_seconds": start,
                        "end_seconds": end,
                        "applied": False,
                        "decision": "diarization_only_after_separator_error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                transition_audits.append(audit)

            speaker_rows: list[dict[str, Any]] = []
            for speaker_index, stem in enumerate(stems):
                peak = float(np.max(np.abs(stem)))
                if peak > 0.95:
                    stem = stem * (0.95 / peak)
                path = run_dir / f"speaker_{speaker_index + 1:02d}.wav"
                _write(path, stem)
                speaker_turns = [
                    {
                        "start_seconds": float(row["start_seconds"]),
                        "end_seconds": float(row["end_seconds"]),
                    }
                    for row in turns
                    if int(row["speaker_index"]) == speaker_index
                ]
                try:
                    asr = _post_json(
                        LIVE_URL,
                        {"audio_path": str(path.resolve()), "verify_speakers": False},
                        timeout=180,
                    )
                    hypothesis = str(asr.get("hypothesis") or "").strip()
                except (OSError, RuntimeError, urllib.error.URLError) as error:
                    hypothesis = ""
                    asr = {"error": f"{type(error).__name__}: {error}"}
                speaker_rows.append(
                    {
                        "speaker_id": f"SPEAKER_{speaker_index + 1:02d}",
                        "display_name": f"Người nói {speaker_index + 1}",
                        "audio_path": str(path.resolve()),
                        "hypothesis": hypothesis,
                        "turns": speaker_turns,
                        "asr_error": asr.get("error"),
                    }
                )

            receipt = {
                "format": "qces_multispeaker_region_live_v3",
                "complete": True,
                "input_path": str(audio_path.resolve()),
                "duration_seconds": duration,
                "speaker_count": speaker_count,
                "speaker_count_is_estimated": True,
                "model_display": "Mixi · Phân cụm nhiều giọng theo vùng",
                "speech_region_seconds": [speech_start, speech_end],
                "event_masked_spans": [
                    {"start_seconds": start, "end_seconds": end, "label": label}
                    for start, end, label in event_spans
                ],
                "diarization": {
                    "window_seconds": window_seconds,
                    "hop_seconds": hop_seconds,
                    "kept_windows": int(keep.sum()),
                    "total_windows": len(starts),
                    "event_masking_fallback": event_masking_fallback,
                    "minimum_unmasked_windows": minimum_unmasked_windows,
                    "count_candidates": count_audit,
                    "selected_speaker_count": speaker_count,
                    "turns": turns,
                },
                "transition_separation": transition_audits,
                "speakers": speaker_rows,
                "warning": (
                    "Số người là ước lượng từ âm sắc. Một giọng đổi âm sắc có thể bị "
                    "chia thành hai cụm; hãy nghe từng stem để kiểm chứng."
                ),
            }
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return receipt


BACKEND: MultiSpeakerRegionBackend


class Handler(BaseHTTPRequestHandler):
    server_version = "QCESMultiSpeakerRegion/1.0"

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "multi-speaker region clustering and local separation",
                    "max_speakers": 4,
                    "max_audio_seconds": MAX_AUDIO_SECONDS,
                    "speaker_count_is_estimated": True,
                },
            )
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/separate":
            self.send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            audio_path = Path(str(payload.get("audio_path") or "")).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            prediction = payload.get("prediction")
            if prediction is not None and not isinstance(prediction, dict):
                raise TypeError("prediction must be an object")
            result = BACKEND.separate(
                audio_path,
                prediction=prediction,
                max_speakers=int(payload.get("max_speakers") or 4),
            )
            self.send_json(200, result)
        except Exception as error:  # noqa: BLE001
            self.send_json(500, {"error": type(error).__name__, "message": str(error)})

    def log_message(self, message: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {message % args}", flush=True)


def main() -> int:
    global BACKEND
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8518)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    BACKEND = MultiSpeakerRegionBackend()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
