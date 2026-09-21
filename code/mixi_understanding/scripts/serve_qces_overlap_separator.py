#!/usr/bin/env python3
"""Serve honest two-speaker separation for the live Mixi Understanding demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "MossFormer2_SS_16K"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / MODEL_NAME
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "qces_overlap_separator_live_v1"
ASR_URL = "http://127.0.0.1:8512/infer"
TWO_SPEAKER_RAW_RMS_RATIO_THRESHOLD = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(b"mossformer2-ss-one-pass-v1")
    return digest.hexdigest()[:20]


def normalize_for_listening(source: Path, destination: Path) -> None:
    waveform, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2) + 1e-12))
    if rms > 1e-6:
        target_rms = 10.0 ** (-20.0 / 20.0)
        waveform = waveform * min(8.0, target_rms / rms)
    peak = float(np.max(np.abs(waveform)) + 1e-12)
    if peak > 0.95:
        waveform = waveform * (0.95 / peak)
    sf.write(destination, waveform.astype(np.float32), sample_rate, subtype="PCM_16")


def post_asr(audio_path: Path) -> dict[str, Any]:
    # Do not ask the live service to call this separator again while this
    # service is already holding its model lock.
    payload = json.dumps(
        {
            "audio_path": str(audio_path.resolve()),
            "verify_speakers": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        ASR_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


class OverlapSeparator:
    def __init__(self) -> None:
        if not (CHECKPOINT_DIR / "last_best_checkpoint.pt").is_file():
            raise FileNotFoundError(CHECKPOINT_DIR / "last_best_checkpoint.pt")
        from clearvoice.network_wrapper import network_wrapper
        from clearvoice.networks import CLS_MossFormer2_SS_16K

        wrapper = network_wrapper()
        wrapper.model_name = MODEL_NAME
        wrapper.load_args_ss()
        wrapper.args.use_cuda = 0
        wrapper.args.task = "speech_separation"
        wrapper.args.checkpoint_dir = str(CHECKPOINT_DIR)
        wrapper.args.output_dir = str(OUTPUT_ROOT)
        wrapper.args.one_time_decode_length = 12
        wrapper.args.decode_window = 8
        self.args = wrapper.args
        self.model = CLS_MossFormer2_SS_16K(self.args)
        self.lock = threading.Lock()

    @staticmethod
    def _verification_audio(
        audio_path: Path,
        speech_segments: list[dict[str, Any]] | None,
    ) -> tuple[np.ndarray, int]:
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        if sample_rate != 16_000:
            from scipy.signal import resample_poly

            divisor = math.gcd(sample_rate, 16_000)
            waveform = resample_poly(
                waveform,
                16_000 // divisor,
                sample_rate // divisor,
            ).astype(np.float32)
            sample_rate = 16_000
        waveform = waveform[: int(round(10.0 * sample_rate))]
        if speech_segments:
            pieces: list[np.ndarray] = []
            for segment in speech_segments:
                start = max(0, int(math.floor(float(segment["start_seconds"]) * sample_rate)))
                end = min(
                    len(waveform),
                    int(math.ceil(float(segment["end_seconds"]) * sample_rate)),
                )
                if end > start:
                    pieces.append(waveform[start:end])
            if pieces:
                waveform = np.concatenate(pieces)
        if len(waveform) < sample_rate:
            waveform = np.pad(waveform, (0, sample_rate - len(waveform)))
        return np.asarray(waveform, dtype=np.float32), sample_rate

    def verify_two_speakers(
        self,
        audio_path: Path,
        speech_segments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Verify a two-speaker candidate before per-stem loudness normalization.

        MossFormer always emits two sources.  Its public decoder normalizes each
        source independently, which can make a tiny artifact from one speaker
        sound like a second talker.  The raw source RMS ratio preserves the
        missing-source signal and is therefore used only as a conservative
        acceptance gate, never as a speaker identity label.
        """

        waveform, sample_rate = self._verification_audio(audio_path, speech_segments)
        duration = len(waveform) / sample_rate
        if duration < 0.5:
            raise ValueError("audio must contain at least 0.5 seconds of speech")

        with self.lock:
            mixture = torch.from_numpy(waveform).unsqueeze(0)
            with torch.inference_mode():
                separated = self.model.model(mixture)

        if not isinstance(separated, (list, tuple)) or len(separated) != 2:
            raise RuntimeError("expected two raw separator outputs")
        raw_rms = [
            float(torch.sqrt(torch.mean(source[0].float() ** 2) + 1e-12).item())
            for source in separated
        ]
        major = max(raw_rms)
        minor = min(raw_rms)
        ratio = minor / max(major, 1e-12)
        accepted = ratio >= TWO_SPEAKER_RAW_RMS_RATIO_THRESHOLD
        return {
            "format": "qces_two_speaker_raw_energy_verification_v1",
            "two_speakers_accepted": accepted,
            "raw_source_rms": raw_rms,
            "minor_major_rms_ratio": ratio,
            "minimum_ratio": TWO_SPEAKER_RAW_RMS_RATIO_THRESHOLD,
            "speech_duration_seconds": duration,
            "decision": "two_speakers" if accepted else "one_speaker",
            "note": (
                "Decision is made on raw separator outputs before independent "
                "listening normalization."
            ),
        }

    def separate(self, audio_path: Path) -> dict[str, Any]:
        digest = sha256_file(audio_path)
        run_dir = OUTPUT_ROOT / digest
        receipt_path = run_dir / "receipt.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if all(Path(row["audio_path"]).is_file() for row in receipt.get("speakers", [])):
                return receipt

        with self.lock:
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if all(Path(row["audio_path"]).is_file() for row in receipt.get("speakers", [])):
                    return receipt

            waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            duration = len(waveform) / max(sample_rate, 1)
            if duration < 0.5:
                raise ValueError("audio must be at least 0.5 seconds")
            if duration > 10.1:
                raise ValueError("live overlap separation supports at most 10 seconds")

            run_dir.mkdir(parents=True, exist_ok=True)
            decode_seconds = max(2, min(10, int(math.ceil(duration))))
            self.args.decode_window = decode_seconds
            self.args.one_time_decode_length = 12
            self.model.process(str(audio_path), online_write=True, output_path=str(run_dir))

            generated = sorted(
                path
                for path in (run_dir / MODEL_NAME).glob("*_s[12].*")
                if path.suffix.lower() in {".wav", ".flac"}
            )
            if len(generated) != 2:
                raise RuntimeError(f"expected two separated stems, found {len(generated)}")

            rows: list[dict[str, Any]] = []
            for source_index, source in enumerate(generated, 1):
                listening_path = run_dir / f"source_{source_index}_listening.wav"
                normalize_for_listening(source, listening_path)
                asr = post_asr(listening_path)
                speech_duration = sum(
                    max(0.0, float(row.get("end_seconds", 0.0)) - float(row.get("start_seconds", 0.0)))
                    for row in asr.get("speech_segments", [])
                )
                rows.append(
                    {
                        "source_index": source_index,
                        "audio_path": str(listening_path.resolve()),
                        "hypothesis": str(asr.get("hypothesis") or "").strip(),
                        "speech_duration_seconds": speech_duration,
                        "speech_segments": asr.get("speech_segments", []),
                    }
                )

            # The dominant/longer voice is presented first. This is a stable demo
            # convention, not a claim about the speakers' real-world identity.
            rows.sort(key=lambda row: (-float(row["speech_duration_seconds"]), row["source_index"]))
            for speaker_number, row in enumerate(rows, 1):
                row["speaker_id"] = f"SPEAKER_{speaker_number:02d}"
                row["display_name"] = f"Người nói {speaker_number}"

            receipt = {
                "format": "qces_two_speaker_overlap_separation_v1",
                "complete": True,
                "input_path": str(audio_path.resolve()),
                "duration_seconds": duration,
                "model_display": "Mixi · Tách hai giọng chồng lấn",
                "speaker_ordering": "dominant speech duration first",
                "speakers": rows,
                "warning": (
                    "Transcript là dự đoán ASR trên từng stem; câu quá nhỏ hoặc bị che mạnh "
                    "có thể vẫn sai và cần nghe lại waveform."
                ),
            }
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return receipt


BACKEND: OverlapSeparator


class Handler(BaseHTTPRequestHandler):
    server_version = "QCESOverlapSeparator/1.0"

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
                    "service": "two-speaker overlap separation",
                    "speaker_verifier": "raw source energy ratio",
                    "minimum_raw_rms_ratio": TWO_SPEAKER_RAW_RMS_RATIO_THRESHOLD,
                },
            )
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/separate", "/verify"}:
            self.send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            audio_path = Path(str(payload.get("audio_path") or "")).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            if self.path == "/verify":
                speech_segments = payload.get("speech_segments")
                if speech_segments is not None and not isinstance(speech_segments, list):
                    raise TypeError("speech_segments must be a list")
                output = BACKEND.verify_two_speakers(audio_path, speech_segments)
            else:
                output = BACKEND.separate(audio_path)
            self.send_json(200, output)
        except Exception as error:  # noqa: BLE001
            self.send_json(500, {"error": type(error).__name__, "message": str(error)})

    def log_message(self, message: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {message % args}", flush=True)


def main() -> int:
    global BACKEND
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8517)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    BACKEND = OverlapSeparator()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
