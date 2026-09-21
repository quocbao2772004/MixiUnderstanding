#!/usr/bin/env python3
"""Route among mixture and enhanced ASR views using Whisper confidence."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve


RATE = 16_000
MODE_SPECS = (
    ("mixture", "audiosep", "mixture_beta_1"),
    ("audiosep_oa", "audiosep", "audiosep_oa_beta_0_25"),
    ("frcrn_oa", "frcrn", "frcrn_oa_beta_0_25"),
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _edit_distance(reference: str, hypothesis: str) -> tuple[int, int]:
    left, right = _words(reference), _words(hypothesis); previous = list(range(len(right) + 1))
    for index, source in enumerate(left, 1):
        current = [index]
        for column, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (source != target)))
        previous = current
    return previous[-1], max(len(left), 1)


def _load_crop(path: Path, start_seconds: float, end_seconds: float) -> np.ndarray:
    wave, source_rate = sf.read(path, dtype="float32", always_2d=True)
    value = torch.from_numpy(wave.mean(axis=1)).float()
    if source_rate != RATE:
        value = AF.resample(value, source_rate, RATE)
    start = max(0, int(math.floor(start_seconds * RATE))); end = min(value.numel(), int(math.ceil(end_seconds * RATE)))
    value = value[start:end]
    rms = value.square().mean().sqrt().clamp_min(1e-8)
    return torch.clamp(value * min(10.0, float((10 ** (-24 / 20)) / rms)), -1.0, 1.0).numpy()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edits = sum(int(row["edit_distance"]) for row in rows); words = sum(int(row["reference_words"]) for row in rows)
    return {"utterances": len(rows), "corpus_wer_↓": edits / max(words, 1), "accuracy_at_wer_0.25_↑": float(np.mean([row["wer_↓"] <= 0.25 for row in rows])), "total_edits": edits, "reference_words": words}


def _feature(row: dict[str, Any]) -> np.ndarray:
    mode = row["mode"]
    return np.asarray([
        1.0, -float(row["mean_token_logprob_↑"]), -float(row["p10_token_logprob_↑"]),
        float(row["generated_tokens"]) / 40.0, float(row["compression_ratio_↔"]) / 2.0,
        float(mode == "audiosep_oa"), float(mode == "frcrn_oa"),
    ], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_chaos_v1")
    parser.add_argument("--audiosep-items", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_chaos_audiosep_oa_v1/asr_items.jsonl")
    parser.add_argument("--frcrn-items", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_chaos_frcrn_oa_v1/asr_items.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_chaos_confidence_router_v1")
    parser.add_argument("--asr-model", default=str(Path.home() / ".cache/huggingface/hub/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    scenes = {row["scene_id"]: row for row in _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")}
    sources = {"audiosep": _jsonl(args.audiosep_items.resolve()), "frcrn": _jsonl(args.frcrn_items.resolve())}
    source_maps = {name: {(row["split"], row["scene_id"], row["mode"]): row for row in rows} for name, rows in sources.items()}
    audio: list[np.ndarray] = []; metadata: list[dict[str, Any]] = []
    for scene_id, scene in sorted(scenes.items(), key=lambda item: (item[1]["split"], item[0])):
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        for mode, source_name, source_mode in MODE_SPECS:
            source = source_maps[source_name][(scene["split"], scene_id, source_mode)]
            path = _resolve(source["audio_path"])
            audio.append(_load_crop(path, max(0.0, float(speech["onset_seconds"]) - 0.15), float(speech["offset_seconds"]) + 0.15))
            metadata.append({"scene_id": scene_id, "split": scene["split"], "difficulty": scene["difficulty"], "measured_snr_db": scene["measured_speech_to_all_interference_snr_db"], "mode": mode, "reference": speech["transcript"], "audio_path": _portable(path)})
    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.asr_model, local_files_only=True, dtype=torch.float16).cuda().eval()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(audio), args.batch_size):
        batch_audio, batch_meta = audio[start:start + args.batch_size], metadata[start:start + args.batch_size]
        features = processor(batch_audio, sampling_rate=RATE, return_tensors="pt").input_features.cuda().half()
        with torch.inference_mode():
            generated = model.generate(features, language="vi", task="transcribe", max_new_tokens=96, no_repeat_ngram_size=3, repetition_penalty=1.05, return_dict_in_generate=True, output_scores=True)
            transition = model.compute_transition_scores(generated.sequences, generated.scores, normalize_logits=True).float().cpu()
        hypotheses = processor.batch_decode(generated.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for meta, hypothesis, scores in zip(batch_meta, hypotheses, transition):
            # Whisper may assign ``-inf`` to forced/suppressed control tokens.
            # They are not acoustic confidence observations and would poison the
            # per-mode calibration and the regression router.
            valid = scores[torch.isfinite(scores) & (scores.abs() > 1e-9)]
            valid = valid if valid.numel() else torch.tensor([-100.0])
            text = hypothesis.strip(); edits, word_count = _edit_distance(meta["reference"], text)
            compressed = len(zlib.compress(text.encode("utf-8")))
            rows.append({**meta, "hypothesis": text, "edit_distance": edits, "reference_words": word_count, "wer_↓": edits / word_count, "mean_token_logprob_↑": float(valid.mean()), "p10_token_logprob_↑": float(torch.quantile(valid, 0.1)), "generated_tokens": int(valid.numel()), "compression_ratio_↔": len(text.encode("utf-8")) / max(compressed, 1)})
        print(json.dumps({"scored": min(start + len(batch_audio), len(audio)), "total": len(audio)}), flush=True)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows: grouped[(row["split"], row["scene_id"])].append(row)
    val_rows = [row for row in rows if row["split"] == "val"]
    mode_stats = {}
    for mode, _, _ in MODE_SPECS:
        values = np.asarray([row["mean_token_logprob_↑"] for row in val_rows if row["mode"] == mode])
        mode_stats[mode] = {"mean": float(values.mean()), "std": float(max(values.std(), 1e-6))}
    x = np.stack([_feature(row) for row in val_rows]); y = np.asarray([min(float(row["wer_↓"]), 2.0) for row in val_rows])
    ridge = 1.0; penalty = np.eye(x.shape[1]); penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + ridge * penalty, x.T @ y)
    methods = ("max_raw_confidence", "max_mode_normalized_confidence", "ridge_expected_wer")
    selected_rows: dict[str, dict[str, list[dict[str, Any]]]] = {method: {"val": [], "test": []} for method in methods}
    oracle_rows: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    fixed_rows: dict[str, dict[str, list[dict[str, Any]]]] = {mode: {"val": [], "test": []} for mode, _, _ in MODE_SPECS}
    for (split, _), candidates in grouped.items():
        for row in candidates: fixed_rows[row["mode"]][split].append(row)
        selected_rows["max_raw_confidence"][split].append(max(candidates, key=lambda row: row["mean_token_logprob_↑"]))
        selected_rows["max_mode_normalized_confidence"][split].append(max(candidates, key=lambda row: (row["mean_token_logprob_↑"] - mode_stats[row["mode"]]["mean"]) / mode_stats[row["mode"]]["std"]))
        selected_rows["ridge_expected_wer"][split].append(min(candidates, key=lambda row: float(_feature(row) @ coefficients)))
        oracle_rows[split].append(min(candidates, key=lambda row: row["edit_distance"]))
    validation = {method: _summary(selected_rows[method]["val"]) for method in methods}
    chosen_method = min(methods, key=lambda method: (validation[method]["corpus_wer_↓"], methods.index(method)))
    summary = {
        "fixed": {mode: {split: _summary(by_split[split]) for split in ("val", "test")} for mode, by_split in fixed_rows.items()},
        "routers": {method: {split: _summary(selected_rows[method][split]) for split in ("val", "test")} for method in methods},
        "selected_on_validation": {"method": chosen_method, "validation": validation[chosen_method], "locked_test": _summary(selected_rows[chosen_method]["test"])},
        "oracle_candidate_analysis_only": {split: _summary(oracle_rows[split]) for split in ("val", "test")},
    }
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    row_path = output / "scored_candidates.jsonl"; _atomic_text(row_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    selection_path = output / "selected_items.jsonl"; selected_output = [{"router": chosen_method, **row} for split in ("val", "test") for row in selected_rows[chosen_method][split]]; _atomic_text(selection_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected_output))
    receipt = {"format": "qces_vietnamese_asr_confidence_router_receipt_v1", "complete": True, "candidate_modes": [mode for mode, _, _ in MODE_SPECS], "confidence_source": "Whisper normalized mean token log-probability", "ground_truth_features_at_inference": False, "selection_protocol": "router family selected on validation corpus WER and locked on test", "mode_confidence_calibration": mode_stats, "ridge_coefficients": coefficients.tolist(), "summary": summary, "scored_candidates": _portable(row_path), "selected_items": _portable(selection_path)}
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
