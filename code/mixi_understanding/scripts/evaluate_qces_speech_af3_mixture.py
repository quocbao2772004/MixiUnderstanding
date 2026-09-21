#!/usr/bin/env python3
"""Evaluate frozen Audio Flamingo 3 on noisy speech-event mixtures.

This evaluator is intentionally separate from the existing multiple-choice
AudioQA audit.  It evaluates only transcript questions from the speech-event
branch, gives AF3 the original mixture plus the original question, and scores
the free-form answer with normalized word error rate (WER).

Rows are appended and fsynced one at a time so an interrupted 8B-model run can
resume without recomputing completed questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import soundfile as sf

from mixi_understanding.scripts.evaluate_qces_audioqa import (
    AudioFlamingo3OptionScorer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--audio-flamingo-3-hf/snapshots"
    / "7d4bae64ee29878af6504ae6f6bb3e40492838ad"
)
PROMPT_VERSION = "qces_speech_af3_exact_transcript_v1"
FORMAT_VERSION = "qces_speech_af3_mixture_eval_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower().replace("'", ""))


def _wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    left, right = _tokens(reference), _tokens(hypothesis)
    if not left:
        errors = len(right)
        return (0.0 if not right else 1.0), errors, 0
    previous = list(range(len(right) + 1))
    for row_index, source in enumerate(left, start=1):
        current = [row_index]
        for column_index, target in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + int(source != target),
                )
            )
        previous = current
    errors = previous[-1]
    return errors / len(left), errors, len(left)


def _prompt(question: str) -> str:
    return (
        "Listen carefully to the supplied noisy audio. "
        "Answer the question by transcribing only the requested spoken utterance. "
        "Do not describe background sounds and do not add an explanation. "
        "Preserve the spoken wording as exactly as possible.\n"
        f"Question: {question}\n"
        "Transcript:"
    )


def _clean_generation(value: str) -> str:
    result = value.strip()
    result = re.sub(r"^(?:transcript|answer)\s*:\s*", "", result, flags=re.I)
    if len(result) >= 2 and result[0] == result[-1] and result[0] in {'"', "'"}:
        result = result[1:-1].strip()
    return result


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else float("nan")


def _metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    errors = sum(int(row["edit_errors"]) for row in rows)
    words = sum(int(row["reference_words"]) for row in rows)
    return {
        "questions": len(rows),
        "utterance_accuracy_at_wer_0.25_↑": _mean(
            float(row["wer"] <= 0.25) for row in rows
        ),
        "utterance_accuracy_at_wer_0.10_↑": _mean(
            float(row["wer"] <= 0.10) for row in rows
        ),
        "mean_utterance_wer_↓": _mean(float(row["wer"]) for row in rows),
        "corpus_wer_↓": float(errors / words) if words else float("nan"),
        "exact_normalized_transcript_accuracy_↑": _mean(
            float(_tokens(str(row["prediction"])) == _tokens(str(row["answer"])))
            for row in rows
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_speech_event_hard_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_speech_event_hard_v2_af3_mixture",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quantization", choices=("4bit", "8bit", "none"), default="4bit")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    item_path = output_dir / "items.jsonl"
    receipt_path = output_dir / "receipt.json"
    if args.overwrite:
        item_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)

    questions = [
        row
        for row in _read_jsonl(dataset_dir / "questions.jsonl")
        if row.get("split") == "test" and row.get("answer_type") == "transcript"
    ]
    questions.sort(key=lambda row: str(row["question_id"]))
    if args.max_records is not None:
        questions = questions[: args.max_records]
    scenes = {
        str(row["scene_id"]): row for row in _read_jsonl(dataset_dir / "scenes.jsonl")
    }

    completed: dict[str, dict[str, Any]] = {}
    if item_path.exists():
        for row in _read_jsonl(item_path):
            completed[str(row["question_id"])] = row
    pending = [row for row in questions if str(row["question_id"]) not in completed]
    print(
        f"[DATA] transcript_test_questions={len(questions)} "
        f"completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    scorer = None
    if pending:
        print(f"[MODEL] loading AF3 from {args.model}", flush=True)
        scorer = AudioFlamingo3OptionScorer(
            model_name=str(args.model.resolve()),
            revision=None,
            quantization=args.quantization,
            dtype=args.dtype,
            device="auto",
            device_map="auto",
            attention_implementation="sdpa",
            local_files_only=True,
            seed=args.seed,
        )
        print("[MODEL] loaded", flush=True)

    for index, question in enumerate(pending, start=1):
        scene = scenes[str(question["scene_id"])]
        audio_path = _resolve(str(question["mixture_path"]))
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = np.ascontiguousarray(waveform.mean(axis=1), dtype=np.float32)
        assert scorer is not None
        raw_prediction = scorer.generate_text(
            _prompt(str(question["question"])),
            mono,
            int(sample_rate),
            max_new_tokens=args.max_new_tokens,
        )
        prediction = _clean_generation(raw_prediction)
        wer, edit_errors, reference_words = _wer(str(question["answer"]), prediction)
        row = {
            "format": FORMAT_VERSION,
            "condition": "af3_plus_original_mixture",
            "prompt_version": PROMPT_VERSION,
            "question_id": question["question_id"],
            "scene_id": question["scene_id"],
            "operation": question["operation"],
            "question": question["question"],
            "answer": question["answer"],
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "wer": wer,
            "edit_errors": edit_errors,
            "reference_words": reference_words,
            "correct_at_wer_0.25": bool(wer <= 0.25),
            "difficulty": scene["difficulty"],
            "requested_speech_to_overlap_noise_snr_db": scene[
                "requested_speech_to_overlap_noise_snr_db"
            ],
            "measured_first_speech_to_overlap_noise_snr_db": scene[
                "measured_first_speech_to_overlap_noise_snr_db"
            ],
            "mixture_path": str(question["mixture_path"]),
            "mixture_sha256": _sha256(audio_path),
        }
        with item_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        completed[str(question["question_id"])] = row
        print(
            f"[ITEM {len(completed)}/{len(questions)}] {question['question_id']} "
            f"tier={scene['difficulty']} wer={wer:.3f} pred={prediction[:90]!r}",
            flush=True,
        )

    rows = [completed[str(row["question_id"])] for row in questions]
    by_difficulty: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_operation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[str(row["difficulty"])].append(row)
        by_operation[str(row["operation"])].append(row)
    receipt = {
        "format": FORMAT_VERSION,
        "complete": len(rows) == len(questions),
        "condition": "AF3 + original noisy mixture + original question",
        "prompt_version": PROMPT_VERSION,
        "dataset_dir": str(dataset_dir.relative_to(PROJECT_ROOT)),
        "model": str(args.model.resolve()),
        "quantization": args.quantization,
        "dtype": args.dtype,
        "metrics": _metrics(rows),
        "metrics_by_difficulty": {
            key: _metrics(value) for key, value in sorted(by_difficulty.items())
        },
        "metrics_by_operation": {
            key: _metrics(value) for key, value in sorted(by_operation.items())
        },
        "items": str(item_path.relative_to(PROJECT_ROOT)),
        "items_sha256": _sha256(item_path),
    }
    _write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
