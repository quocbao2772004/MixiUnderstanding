#!/usr/bin/env python3
"""Separate audible QA evidence and residual with a trained QCES mask model."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from mixi_understanding.qces.composer import ROLE_NAMES
from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    DUAL_ROLE_SEMANTIC_MODE,
    LEGACY_TEMPORAL_ROLE_MODE,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
)
from mixi_understanding.qces.model import load_qces_checkpoint
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    CANONICAL_NUM_SAMPLES,
    CANONICAL_SAMPLE_RATE,
    SampleSpec,
    configure_determinism as configure_clap_determinism,
    encode_canonical_waveform,
    encode_full_questions,
    file_identity,
    load_frozen_encoder,
    source_tree_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path)
    parser.add_argument("--audiosep-config", type=Path)
    parser.add_argument("--audiosep-checkpoint", type=Path)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--window-start-seconds",
        type=float,
        default=0.0,
        help=(
            "start of the canonical 10 s model window; shorter tails are "
            "right-padded with silence"
        ),
    )
    parser.add_argument(
        "--foundation-seed",
        type=int,
        default=2026,
        help="deterministic seed for online frozen AudioSep-CLAP extraction",
    )
    return parser.parse_args()


def prepare_audio_window(
    path: Path,
    target_rate: int,
    *,
    start_seconds: float = 0.0,
    window_seconds: float = 10.0,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if not np.isfinite(start_seconds) or start_seconds < 0.0:
        raise ValueError("window start must be a finite non-negative value")
    if not np.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError("window length must be finite and positive")
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[0] == 0:
        raise ValueError("input audio is empty")
    original_channels = int(waveform.shape[1])
    original_samples = int(waveform.shape[0])
    mono = waveform.mean(axis=1)
    if not np.isfinite(mono).all():
        raise ValueError("input audio contains NaN or Inf")
    if sample_rate != target_rate:
        divisor = int(np.gcd(sample_rate, target_rate))
        mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor)
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    start_sample = int(round(start_seconds * target_rate))
    if start_sample >= mono.shape[0]:
        raise ValueError(
            "window start lies beyond the resampled input duration "
            f"({mono.shape[0] / target_rate:.3f} s)"
        )
    target_samples = int(round(window_seconds * target_rate))
    available = mono[start_sample : start_sample + target_samples]
    copied_samples = int(available.shape[0])
    result = np.zeros(target_samples, dtype=np.float32)
    result[:copied_samples] = available
    preprocessing = {
        "original_sample_rate_hz": int(sample_rate),
        "original_num_channels": original_channels,
        "original_num_samples": original_samples,
        "original_duration_seconds": float(original_samples / sample_rate),
        "downmix": "arithmetic_channel_mean",
        "resampled_sample_rate_hz": int(target_rate),
        "window_start_seconds": float(start_seconds),
        "window_duration_seconds": float(window_seconds),
        "window_num_samples": target_samples,
        "copied_input_samples": copied_samples,
        "right_padding_samples": target_samples - copied_samples,
        "tail_truncated": bool(start_sample + target_samples < mono.shape[0]),
    }
    return result, preprocessing


def _read_audio(path: Path, target_rate: int) -> np.ndarray:
    """Backward-compatible canonical-window reader."""

    waveform, _ = prepare_audio_window(path, target_rate)
    return waveform


def _intervals(
    active: torch.Tensor, hop_samples: int, sample_rate: int, max_seconds: float
) -> List[List[float]]:
    active = active.to(torch.bool).cpu().tolist()
    result: List[List[float]] = []
    start = None
    for index, value in enumerate(active + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append(
                [
                    round(start * hop_samples / sample_rate, 6),
                    round(min(index * hop_samples / sample_rate, max_seconds), 6),
                ]
            )
            start = None
    return result


def _uses_audiosep_clap(checkpoint: object) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    config = checkpoint.get("config")
    return (
        isinstance(config, dict)
        and config.get("foundation_feature_mode", "none")
        == AUDIOSEP_CLAP_FOUNDATION_FEATURES
    )


def require_online_foundation_support(
    checkpoint: object,
    *,
    audiosep_root: Path | None = None,
    audiosep_config: Path | None = None,
    audiosep_checkpoint: Path | None = None,
) -> bool:
    """Require the assets needed to reproduce frozen CLAP features online."""

    enabled = _uses_audiosep_clap(checkpoint)
    if enabled and any(
        value is None for value in (audiosep_root, audiosep_config, audiosep_checkpoint)
    ):
        raise SystemExit(
            "audiosep_clap inference requires --audiosep-root, "
            "--audiosep-config, and --audiosep-checkpoint so the frozen online "
            "extractor matches the separation backbone"
        )
    return enabled


def _same_file_identity(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return before.get("sha256") == after.get("sha256") and before.get(
        "size_bytes"
    ) == after.get("size_bytes")


def extract_online_foundation_features(
    *,
    question: str,
    waveform: np.ndarray,
    sample_rate: int,
    audiosep_root: Path,
    audiosep_checkpoint: Path,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Reproduce the cache feature recipe from inference-time inputs only."""

    if sample_rate != CANONICAL_SAMPLE_RATE:
        raise RuntimeError(
            "audiosep_clap inference requires the model sample rate to be "
            f"{CANONICAL_SAMPLE_RATE}, got {sample_rate}"
        )
    if waveform.shape != (CANONICAL_NUM_SAMPLES,):
        raise RuntimeError(
            "audiosep_clap inference is aligned to one canonical 10-second "
            f"clip ({CANONICAL_NUM_SAMPLES} samples), got {waveform.shape}"
        )
    if waveform.dtype != np.float32 or not np.isfinite(waveform).all():
        raise RuntimeError("audiosep_clap inference requires a finite float32 waveform")
    if not question.strip():
        raise RuntimeError("audiosep_clap inference requires a non-empty question")

    checkpoint_before = file_identity(audiosep_checkpoint)
    source_before = source_tree_identity(audiosep_root)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    determinism = configure_clap_determinism(device, seed)
    encoder, state_provenance = load_frozen_encoder(
        audiosep_root,
        audiosep_checkpoint,
        device,
    )
    try:
        sample = SampleSpec(
            sample_id="online_sample",
            scene_id="online_scene",
            question=question,
        )
        question_feature = encode_full_questions(encoder, [sample], batch_size=1)[
            sample.sample_id
        ]
        scene_feature = encode_canonical_waveform(
            encoder,
            waveform,
            scene_id=sample.scene_id,
            device=device,
        )
    finally:
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    checkpoint_after = file_identity(audiosep_checkpoint)
    source_after = source_tree_identity(audiosep_root)
    if not _same_file_identity(checkpoint_before, checkpoint_after):
        raise RuntimeError("AudioSep checkpoint changed during online extraction")
    if source_before != source_after:
        raise RuntimeError("AudioSep source tree changed during online extraction")
    provenance = {
        "mode": AUDIOSEP_CLAP_FOUNDATION_FEATURES,
        "source": "online_frozen_audiosep_clap",
        "question_shape": [512],
        "scene_shape": [32, 512],
        "canonical_num_samples": CANONICAL_NUM_SAMPLES,
        "audiosep_checkpoint": checkpoint_before,
        "audiosep_source_tree": source_before,
        "query_encoder_state": state_provenance,
        "determinism": determinism,
        "contains_event_answer_or_oracle_inputs": False,
    }
    return (
        question_feature.unsqueeze(0).to(device),
        scene_feature.unsqueeze(0).to(device),
        provenance,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise SystemExit("threshold must lie strictly between 0 and 1")
    if not np.isfinite(args.window_start_seconds) or args.window_start_seconds < 0:
        raise SystemExit("window-start-seconds must be finite and non-negative")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    foundation_enabled = require_online_foundation_support(
        checkpoint,
        audiosep_root=args.audiosep_root,
        audiosep_config=args.audiosep_config,
        audiosep_checkpoint=args.audiosep_checkpoint,
    )
    config_payload = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if not isinstance(config_payload, dict):
        raise SystemExit("QCES checkpoint is missing its model config")
    sample_rate = config_payload.get("sample_rate", CANONICAL_SAMPLE_RATE)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise SystemExit("QCES checkpoint has an invalid sample rate")
    try:
        waveform, audio_preprocessing = prepare_audio_window(
            args.audio.resolve(),
            sample_rate,
            start_seconds=args.window_start_seconds,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"input audio preparation failed: {exc}") from exc
    question_clap = None
    scene_clap = None
    foundation_provenance = None
    if foundation_enabled:
        assert args.audiosep_root is not None
        assert args.audiosep_checkpoint is not None
        try:
            question_clap, scene_clap, foundation_provenance = (
                extract_online_foundation_features(
                    question=args.question,
                    waveform=waveform,
                    sample_rate=sample_rate,
                    audiosep_root=args.audiosep_root.resolve(),
                    audiosep_checkpoint=args.audiosep_checkpoint.resolve(),
                    device=device,
                    seed=args.foundation_seed,
                )
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"online AudioSep-CLAP extraction failed: {exc}") from exc
    model = load_qces_checkpoint(
        checkpoint,
        map_location=device,
        audiosep_repository_root=(
            str(args.audiosep_root.resolve()) if args.audiosep_root else None
        ),
        audiosep_config_path=(
            str(args.audiosep_config.resolve()) if args.audiosep_config else None
        ),
        audiosep_checkpoint_path=(
            str(args.audiosep_checkpoint.resolve())
            if args.audiosep_checkpoint
            else None
        ),
    ).eval()
    mixture = torch.from_numpy(waveform)[None].to(device)
    tokenizer = StableHashTokenizer(
        model.config.vocab_size, model.config.max_question_tokens
    )
    with torch.inference_mode():
        output = model.forward_questions(
            mixture,
            [args.question],
            tokenizer,
            question_clap=question_clap,
            scene_clap=scene_clap,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persist the exact resampled/cropped/padded window seen by QCES.  The demo
    # captions and replays this file rather than a potentially longer upload.
    sf.write(output_dir / "mixture.wav", waveform, model.config.sample_rate)
    evidence = output.evidence[0].cpu().numpy()
    residual = output.residual[0].cpu().numpy()
    sf.write(output_dir / "evidence.wav", evidence, model.config.sample_rate)
    sf.write(output_dir / "residual.wav", residual, model.config.sample_rate)
    if output.separation.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE:
        if (
            output.separation.anchor_raw_evidence is None
            or output.separation.answer_raw_evidence is None
        ):
            raise RuntimeError("dual_role inference lacks role-specific raw stems")
        sf.write(
            output_dir / "diagnostic_anchor_raw.wav",
            output.separation.anchor_raw_evidence[0].cpu().numpy(),
            model.config.sample_rate,
        )
        sf.write(
            output_dir / "diagnostic_answer_raw.wav",
            output.separation.answer_raw_evidence[0].cpu().numpy(),
            model.config.sample_rate,
        )

    temporal_role_mode = output.composition.temporal_role_mode
    role_probabilities = output.composition.role_probabilities[0]
    if temporal_role_mode == LEGACY_TEMPORAL_ROLE_MODE:
        role_ids = output.composition.role_logits[0].argmax(dim=-1)
        role_activity = {
            role_id: role_ids == role_id for role_id in range(1, len(ROLE_NAMES))
        }
    elif temporal_role_mode == OVERLAP_AWARE_TEMPORAL_ROLE_MODE:
        # Independent thresholds intentionally permit anchor and answer
        # intervals to overlap in the exported grounding.
        role_activity = {
            role_id: role_probabilities[:, role_id] >= args.threshold
            for role_id in range(1, len(ROLE_NAMES))
        }
    else:
        raise RuntimeError(f"unsupported temporal role mode: {temporal_role_mode}")
    duration = len(waveform) / model.config.sample_rate
    roles: Dict[str, List[List[float]]] = {}
    for role_id, name in enumerate(ROLE_NAMES[1:], start=1):
        roles[name] = _intervals(
            role_activity[role_id],
            output.composition.frame_hop_samples,
            model.config.sample_rate,
            duration,
        )
    metadata = {
        "question": args.question,
        "sample_rate": model.config.sample_rate,
        "duration_seconds": duration,
        "audio_preprocessing": audio_preprocessing,
        "foundation_features": foundation_provenance,
        "temporal_role_mode": temporal_role_mode,
        "semantic_separation_mode": output.separation.semantic_separation_mode,
        "foundation_semantic_mixing_mode": (
            model.config.foundation_semantic_mixing_mode
        ),
        "foundation_semantic_candidate_weight_descriptive": (
            float(output.composition.foundation_semantic_candidate_weight.cpu())
            if output.composition.foundation_semantic_candidate_weight is not None
            else None
        ),
        "physical_separator_forwards_per_batch": (
            output.separation.physical_separator_forwards_per_batch
        ),
        "effective_separator_evaluations_per_record": (
            output.separation.effective_separator_evaluations_per_record
        ),
        "same_semantic_probability": (
            float(output.composition.same_semantic_probability[0].cpu())
            if output.composition.same_semantic_probability is not None
            else None
        ),
        "same_semantic_routing": (
            "learned differentiable shared-target interpolation; no label "
            "metadata or thresholded exact reuse at inference"
            if output.separation.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
            else None
        ),
        "no_evidence_probability": float(
            output.composition.no_evidence_logit[0].sigmoid().cpu()
        ),
        "role_intervals": roles,
        "mixture_consistency_l1": float(output.separation.mixture_error[0].cpu()),
    }
    (output_dir / "grounding.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
