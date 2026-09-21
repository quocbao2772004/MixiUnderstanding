#!/usr/bin/env python3
"""Predict one multiple-choice answer from an evidence WAV with frozen AF3.

This single-item inference entry point deliberately accepts no gold answer.  It
is intended for the interactive QCES demo, while the paired, gold-aware audit
remains in ``evaluate_qces_audioqa.py``.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf

from mixi_understanding.scripts.evaluate_qces_audioqa import (
    OPTION_LABELS,
    PROMPT_VERSION,
    SCORING_VERSION,
    AudioFlamingo3OptionScorer,
)


FORMAT_VERSION = "qces_single_audioqa_inference_v2"
SCENE_INVENTORY_PROMPT_VERSION = "qces_demo_scene_inventory_v1"
SCENE_INVENTORY_PROMPT = """Listen to the supplied audio window and list the distinct audible sound-event occurrences in chronological onset order. Include repeated occurrences and background sounds when audible. Do not answer any question. Return only JSON with this exact shape: {"events":[{"order":1,"sound":"short sound description"}]}."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--mixture-audio",
        type=Path,
        help=(
            "Optional exact mixture window used to generate a display-only, "
            "predicted audible-event inventory with the already-loaded AF3."
        ),
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--option", action="append", dest="options", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--quantization", choices=("none", "4bit", "8bit"), default="4bit"
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        args.question, args.options = validate_request(args.question, args.options)
    except ValueError as error:
        parser.error(str(error))
    local_model = Path(args.model).expanduser().exists()
    if not local_model and args.revision is None:
        parser.error("a remote --model requires a pinned --revision")
    return args


def validate_request(
    question: str, options: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    question = question.strip()
    normalized = tuple(option.strip() for option in options)
    if not question:
        raise ValueError("question must not be empty")
    if not 2 <= len(normalized) <= len(OPTION_LABELS):
        raise ValueError("provide between two and five --option values")
    if any(not option for option in normalized):
        raise ValueError("answer options must not be empty")
    folded = [option.casefold() for option in normalized]
    if len(set(folded)) != len(folded):
        raise ValueError("answer options must be distinct (case-insensitive)")
    return question, normalized


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[0] == 0:
        raise ValueError("audio is empty")
    mono = np.ascontiguousarray(waveform.mean(axis=1), dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError("audio contains NaN or Inf")
    return mono, int(sample_rate)


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _first_json_object(text: str) -> Mapping[str, Any] | None:
    """Extract one balanced JSON object without accepting trailing prose."""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, Mapping) else None
    return None


def parse_scene_inventory(raw: str) -> dict[str, Any]:
    """Normalize strict AF3 inventory JSON for a fail-visible demo card."""

    payload = _first_json_object(raw)
    if payload is None or set(payload) != {"events"}:
        return {"status": "invalid_json", "events": [], "raw_text": raw.strip()}
    events = payload.get("events")
    if not isinstance(events, list) or not events or len(events) > 32:
        return {"status": "invalid_schema", "events": [], "raw_text": raw.strip()}
    normalized: list[dict[str, Any]] = []
    for expected_order, event in enumerate(events, 1):
        if not isinstance(event, Mapping) or set(event) != {"order", "sound"}:
            return {
                "status": "invalid_schema",
                "events": [],
                "raw_text": raw.strip(),
            }
        sound = event.get("sound")
        if event.get("order") != expected_order or not isinstance(sound, str):
            return {
                "status": "invalid_schema",
                "events": [],
                "raw_text": raw.strip(),
            }
        cleaned = " ".join(sound.strip().split()).strip("`\"'")
        if not cleaned or len(cleaned) > 240 or any(char in cleaned for char in "{}[]"):
            return {
                "status": "invalid_schema",
                "events": [],
                "raw_text": raw.strip(),
            }
        normalized.append({"order": expected_order, "sound": cleaned})
    return {"status": "valid", "events": normalized, "raw_text": None}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        raise SystemExit(f"audio does not exist: {audio_path}")
    try:
        waveform, sample_rate = read_audio(audio_path)
        scorer = AudioFlamingo3OptionScorer(
            model_name=args.model,
            revision=args.revision,
            quantization=args.quantization,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
            attention_implementation=args.attention_implementation,
            local_files_only=args.local_files_only,
            seed=args.seed,
        )
        score = scorer.score(args.question, args.options, waveform, sample_rate)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"AudioQA inference failed: {error}") from error

    scene_inventory: dict[str, Any] | None = None
    mixture_identity: dict[str, Any] | None = None
    if args.mixture_audio is not None:
        mixture_path = args.mixture_audio.expanduser().resolve()
        if not mixture_path.is_file():
            scene_inventory = {
                "status": "caption_error",
                "events": [],
                "raw_text": None,
                "error": f"mixture audio does not exist: {mixture_path}",
            }
        else:
            try:
                mixture_waveform, mixture_rate = read_audio(mixture_path)
                raw_inventory = scorer.generate_text(
                    SCENE_INVENTORY_PROMPT,
                    mixture_waveform,
                    mixture_rate,
                    max_new_tokens=160,
                )
                scene_inventory = parse_scene_inventory(raw_inventory)
                mixture_identity = {
                    "path": str(mixture_path),
                    "sha256": sha256_file(mixture_path),
                    "sample_rate_hz": mixture_rate,
                    "num_samples": int(mixture_waveform.shape[0]),
                    "duration_seconds": float(
                        mixture_waveform.shape[0] / mixture_rate
                    ),
                }
            except (OSError, RuntimeError, ValueError) as error:
                scene_inventory = {
                    "status": "caption_error",
                    "events": [],
                    "raw_text": None,
                    "error": str(error),
                }
        scene_inventory.update(
            {
                "source": "frozen_af3_prediction_on_exact_qces_mixture_window",
                "prompt_version": SCENE_INVENTORY_PROMPT_VERSION,
                "not_ground_truth": True,
            }
        )

    labels = OPTION_LABELS[: len(args.options)]
    rows = [
        {
            "index": index,
            "label": labels[index],
            "option": option,
            "log_probability_↑": float(score.log_scores[index]),
            "probability_↑": float(score.probabilities[index]),
        }
        for index, option in enumerate(args.options)
    ]
    result = {
        "format": FORMAT_VERSION,
        "question": args.question,
        "options": rows,
        "prediction": {
            "index": score.predicted_index,
            "label": labels[score.predicted_index],
            "answer": args.options[score.predicted_index],
            "confidence_↑": float(score.probabilities[score.predicted_index]),
        },
        "audio": {
            "path": str(audio_path),
            "sha256": sha256_file(audio_path),
            "sample_rate_hz": sample_rate,
            "num_samples": int(waveform.shape[0]),
            "duration_seconds": float(waveform.shape[0] / sample_rate),
            "input_channels_downmixed_to_mono": True,
        },
        "scene_inventory": scene_inventory,
        "mixture_audio": mixture_identity,
        "scoring": {
            "method": score.scoring_method,
            "candidate_token_lengths": list(score.candidate_token_lengths),
            "prompt_version": PROMPT_VERSION,
            "scoring_version": SCORING_VERSION,
            "model_provenance": dict(scorer.provenance()),
            "contains_gold_answer_input": False,
        },
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
