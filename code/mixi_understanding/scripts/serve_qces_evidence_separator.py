#!/usr/bin/env python3
"""Persistent A/B evidence-separation service for the public AudioQA demo.

The service never reads QA annotations.  It receives only the live mixture,
the predicted temporal span, and the predicted semantic labels.  Both methods
therefore operate on exactly the same evidence request:

``audiosep``
    Frozen AudioSep with canonical on-manifold text prompts, followed by the
    predicted temporal crop and target-free amplitude projection.

``ours``
    The source-disjoint QCES public-22 TF residual refiner.  It interpolates
    conservatively between the detector crop and the projected AudioSep stem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    encode_prompts,
)
from mixi_understanding.scripts.evaluate_qces_public22_audiosep_postcrop_v1 import (
    AUDIOSEP_RATE,
    PROMPTS,
)
from mixi_understanding.scripts.infer_qces_combined_honest_v1 import load_audio
from mixi_understanding.scripts.train_qces_public22_identity_tf_refiner_v1 import (
    IdentityTFRefiner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RATE = 16_000
CONTEXT_SECONDS = 0.15
MAX_AUDIO_SECONDS = 10.0
METHODS = {"audiosep", "ours"}
OUTPUT_ROOT = PROJECT_ROOT / "outputs/qces_live_evidence_separator"
LIVE_AUDIO_ROOT = PROJECT_ROOT / "outputs/qces_live_recordings"
LABEL_EMBEDDINGS = PROJECT_ROOT / "outputs/qces_label_clap_public22_v1.pt"
REFINER_CHECKPOINT = PROJECT_ROOT / "outputs/qces_public22_identity_tf_refiner_v1/best.pt"
AUDIOSEP_ROOT = PROJECT_ROOT / "code/baseline/audiosep"
AUDIOSEP_CONFIG = AUDIOSEP_ROOT / "config/audiosep_base.yaml"
AUDIOSEP_CHECKPOINT = AUDIOSEP_ROOT / "checkpoint/hf_audiosep/pytorch_model.bin"


def _unique_labels(values: Sequence[Any]) -> list[str]:
    labels: list[str] = []
    for value in values:
        label = str(value)
        if label in PROMPTS and label not in labels:
            labels.append(label)
    return labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EvidenceBackend:
    def __init__(self, device_name: str) -> None:
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        required = [
            LABEL_EMBEDDINGS,
            REFINER_CHECKPOINT,
            AUDIOSEP_CONFIG,
            AUDIOSEP_CHECKPOINT,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing evidence checkpoints: " + ", ".join(missing))

        label_payload = torch.load(LABEL_EMBEDDINGS, map_location="cpu", weights_only=True)
        self.labels = [str(value) for value in label_payload["labels"]]
        self.label_to_id = {label: index for index, label in enumerate(self.labels)}
        if set(self.labels) != set(PROMPTS):
            raise RuntimeError("public-22 label/prompt mismatch")

        self.prompt_embeddings = encode_prompts(
            AUDIOSEP_ROOT,
            AUDIOSEP_CHECKPOINT,
            [PROMPTS[label] for label in self.labels],
            batch_size=len(self.labels),
        )
        self.separator = _load_separator(
            SimpleNamespace(
                audiosep_root=AUDIOSEP_ROOT,
                audiosep_config=AUDIOSEP_CONFIG,
                audiosep_checkpoint=AUDIOSEP_CHECKPOINT,
            ),
            self.device,
        )
        checkpoint = torch.load(REFINER_CHECKPOINT, map_location="cpu", weights_only=True)
        self.refiner = IdentityTFRefiner(num_labels=int(checkpoint["num_labels"]))
        self.refiner.load_state_dict(checkpoint["model_state"])
        self.refiner.to(self.device).eval()
        self.lock = threading.Lock()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        print(
            f"[ready] AudioSep + QCES TF refiner on {self.device}; {len(self.labels)} labels",
            flush=True,
        )

    def _validate_path(self, path: Path) -> Path:
        resolved = path.resolve()
        live_root = LIVE_AUDIO_ROOT.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if live_root not in resolved.parents:
            raise PermissionError("audio_path must be a materialized live-demo recording")
        return resolved

    @torch.inference_mode()
    def _audiosep_candidates(
        self,
        mixture: torch.Tensor,
        condition: torch.Tensor,
        labels: Sequence[str],
    ) -> tuple[list[torch.Tensor], list[float]]:
        mixture_32k = AF.resample(mixture, RATE, AUDIOSEP_RATE)
        batch = mixture_32k.expand(len(labels), -1)
        embeddings = torch.stack(
            [self.prompt_embeddings[PROMPTS[label]] for label in labels]
        ).to(self.device)
        raw_32k = self.separator(
            {"mixture": batch[:, None], "condition": embeddings}
        )["waveform"][:, 0]
        raw = AF.resample(raw_32k, AUDIOSEP_RATE, RATE)
        if raw.shape[-1] != mixture.shape[-1]:
            raw = F.interpolate(
                raw[:, None], size=mixture.shape[-1], mode="linear", align_corners=False
            )[:, 0]
        crop = mixture * condition
        projected: list[torch.Tensor] = []
        scales: list[float] = []
        for candidate in raw:
            postcrop = candidate[None] * condition
            scale = (
                (postcrop * crop).sum(-1, keepdim=True)
                / postcrop.square().sum(-1, keepdim=True).clamp_min(1e-8)
            ).clamp(0.25, 4.0)
            projected.append((postcrop * scale)[0])
            scales.append(float(scale[0, 0].detach().cpu()))
        return projected, scales

    @torch.inference_mode()
    def separate(
        self,
        audio_path: Path,
        start_seconds: float,
        end_seconds: float,
        labels: Sequence[Any],
        method: str,
    ) -> dict[str, Any]:
        audio_path = self._validate_path(audio_path)
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        target_labels = _unique_labels(labels)
        if not target_labels:
            raise ValueError("no supported evidence labels")

        waveform, sample_rate = load_audio(audio_path, RATE)
        duration = min(len(waveform) / sample_rate, MAX_AUDIO_SECONDS)
        start = max(0.0, min(float(start_seconds), duration))
        end = max(start + 0.02, min(float(end_seconds), duration))
        context_start = max(0.0, start - CONTEXT_SECONDS)
        context_end = min(duration, end + CONTEXT_SECONDS)
        left = int(round(context_start * RATE))
        right = int(round(context_end * RATE))
        local = torch.from_numpy(waveform[left:right]).float().to(self.device)[None]
        if local.shape[-1] < int(0.25 * RATE):
            local = F.pad(local, (0, int(0.25 * RATE) - local.shape[-1]))
        condition = torch.zeros_like(local)
        active_left = max(0, int(round((start - context_start) * RATE)))
        active_right = min(local.shape[-1], int(round((end - context_start) * RATE)))
        condition[:, active_left:max(active_left + 1, active_right)] = 1.0

        request_key = json.dumps(
            {
                "audio_sha256": _sha256(audio_path),
                "start": round(start, 5),
                "end": round(end, 5),
                "labels": target_labels,
                "method": method,
                "checkpoint": _sha256(
                    REFINER_CHECKPOINT if method == "ours" else AUDIOSEP_CHECKPOINT
                ),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(request_key.encode("utf-8")).hexdigest()[:24]
        output_dir = OUTPUT_ROOT / digest
        output_path = output_dir / "evidence.wav"
        metadata_path = output_dir / "metadata.json"
        if output_path.is_file() and metadata_path.is_file():
            return json.loads(metadata_path.read_text(encoding="utf-8"))

        with self.lock:
            candidates, projection_scales = self._audiosep_candidates(
                local, condition, target_labels
            )
            crop = local * condition
            if method == "audiosep":
                # Summing is the correct union operation for independently
                # separated target sources.  Peak clipping below only protects
                # browser playback; it does not use a target or oracle signal.
                output = torch.stack(candidates).sum(0)
                mean_gate = None
            else:
                predictions: list[torch.Tensor] = []
                gates: list[float] = []
                for label, candidate in zip(target_labels, candidates):
                    label_id = torch.tensor([self.label_to_id[label]], device=self.device)
                    prediction, gate = self.refiner(crop, candidate[None], label_id)
                    predictions.append(prediction[0])
                    gates.append(float(gate.mean().detach().cpu()))
                # Every per-label refiner already contains the identity crop.
                # Averaging preserves that identity once while combining the
                # semantic edits; summing would duplicate the observed mixture.
                output = torch.stack(predictions).mean(0)
                mean_gate = float(np.mean(gates))

        exact = output[active_left:active_right].detach().float().cpu().numpy()
        peak = float(np.max(np.abs(exact))) if exact.size else 0.0
        if peak > 0.99:
            exact = exact * (0.99 / peak)
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, exact, RATE, subtype="PCM_16")
        metadata: dict[str, Any] = {
            "ok": True,
            "format": "qces_live_evidence_ab_v1",
            "method": method,
            "output_path": str(output_path.resolve()),
            "start_seconds": start,
            "end_seconds": end,
            "labels": target_labels,
            "prompts": [PROMPTS[label] for label in target_labels],
            "projection_scales": projection_scales,
            "mean_tf_gate": mean_gate,
            "annotations_used": False,
            "fallback_used": False,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return metadata


BACKEND: EvidenceBackend


class Handler(BaseHTTPRequestHandler):
    server_version = "QCESEvidenceSeparator/1.0"

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
                    "methods": sorted(METHODS),
                    "event_class_count": len(BACKEND.labels),
                    "annotations_used": False,
                },
            )
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/separate":
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            output = BACKEND.separate(
                Path(str(payload.get("audio_path") or "")),
                float(payload.get("start_seconds", 0.0)),
                float(payload.get("end_seconds", 0.0)),
                list(payload.get("labels") or []),
                str(payload.get("method") or "ours"),
            )
            self._send(200, output)
        except Exception as error:  # noqa: BLE001
            self._send(500, {"error": type(error).__name__, "message": str(error)})

    def log_message(self, message: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {message % args}", flush=True)


def main() -> int:
    global BACKEND
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8516)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    BACKEND = EvidenceBackend(args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
