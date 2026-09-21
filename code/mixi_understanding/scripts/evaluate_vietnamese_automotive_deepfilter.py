#!/usr/bin/env python3
"""Evaluate DeepFilterNet alone and after AudioSep on automotive mixtures."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve


@dataclass
class _AudioMetaData:
    sample_rate: int = 0
    num_frames: int = 0
    num_channels: int = 1
    bits_per_sample: int = 0
    encoding: str = ""


def _install_torchaudio_legacy_import_shim() -> None:
    """DeepFilterNet 0.5.6 imports a metadata class removed by torchaudio 2.11."""
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = _AudioMetaData
    backend.common = common
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules.setdefault("torchaudio.backend.common", common)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load(path: Path, target_rate: int) -> torch.Tensor:
    value, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(value.mean(axis=1)).float()
    return AF.resample(wave, rate, target_rate) if rate != target_rate else wave


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _wer(reference: str, hypothesis: str) -> float:
    left, right = _words(reference), _words(hypothesis)
    previous = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target)))
        previous = current
    return previous[-1] / max(len(left), 1)


def _asr_audio(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-6:
        return value
    gain = min(10.0, (10.0 ** (-24.0 / 20.0)) / rms)
    return np.clip(value * gain, -1.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--audiosep-si-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_audiosep_v1")
    parser.add_argument("--audiosep-asr-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_audiosep_human_v1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_deepfilter_v1")
    parser.add_argument("--asr-model", default=str(Path.home() / ".cache/huggingface/hub/models--vinai--PhoWhisper-tiny/snapshots/cc51d32be916efebde04ff549854fa1741cb5c02"))
    parser.add_argument("--post-filter", action="store_true")
    parser.add_argument("--asr-batch-size", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")

    _install_torchaudio_legacy_import_shim()
    from df import enhance, init_df
    from df.model import ModelParams
    model, df_state, _ = init_df("DeepFilterNet3", post_filter=args.post_filter, log_level="WARNING", log_file=None)
    df_rate = int(ModelParams().sr)
    enhanced_rows = []
    source_modes = {
        "deepfilter_mixture": lambda scene: _resolve(scene["mixture_path"]),
        "audiosep_si_deepfilter": lambda scene: args.audiosep_si_dir.resolve() / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
        "audiosep_asr_deepfilter": lambda scene: args.audiosep_asr_dir.resolve() / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
    }
    for scene in scenes:
        speech = next(x for x in scene["events"] if x["label"] == "Speech")
        target = _load(_resolve(speech["stem_path"]), df_rate)
        mixture = _load(_resolve(scene["mixture_path"]), df_rate)
        mix_len = min(len(mixture), len(target))
        mixture_si = float(scale_invariant_sdr(mixture[:mix_len][None], target[:mix_len][None])[0])
        for mode, source_fn in source_modes.items():
            source = _load(source_fn(scene), df_rate)
            with torch.inference_mode():
                prediction = enhance(model, df_state, source[None], pad=True).detach().cpu()[0]
            path = output / "audio" / mode / scene["split"] / f"{scene['scene_id']}.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, prediction.numpy(), df_rate, format="FLAC", subtype="PCM_16")
            length = min(len(prediction), len(target), len(mixture))
            si = float(scale_invariant_sdr(prediction[:length][None], target[:length][None])[0])
            enhanced_rows.append({
                "scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
                "mode": mode, "path": _portable(path), "si_sdr_db_↑": si,
                "mixture_si_sdr_db_↑": mixture_si, "si_sdri_db_↑": si - mixture_si,
            })
            print(json.dumps({"scene": scene["scene_id"], "mode": mode, "si_sdri": si - mixture_si}), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.asr_model, local_files_only=True, dtype=dtype)
    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
    asr = pipeline("automatic-speech-recognition", model=asr_model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, device=0 if torch.cuda.is_available() else -1)
    deployable_modes = (
        "mixture_full", "deepfilter_mixture", "audiosep_si", "audiosep_si_deepfilter",
        "audiosep_asr", "audiosep_asr_deepfilter",
    )
    audio, metadata = [], []
    row_by_key = {(x["scene_id"], x["mode"]): x for x in enhanced_rows}
    for scene in scenes:
        speech = next(x for x in scene["events"] if x["label"] == "Speech")
        paths = {
            "mixture_full": _resolve(scene["mixture_path"]),
            "deepfilter_mixture": _resolve(row_by_key[(scene["scene_id"], "deepfilter_mixture")]["path"]),
            "audiosep_si": args.audiosep_si_dir.resolve() / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
            "audiosep_si_deepfilter": _resolve(row_by_key[(scene["scene_id"], "audiosep_si_deepfilter")]["path"]),
            "audiosep_asr": args.audiosep_asr_dir.resolve() / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
            "audiosep_asr_deepfilter": _resolve(row_by_key[(scene["scene_id"], "audiosep_asr_deepfilter")]["path"]),
            "clean_speech_upper_bound": _resolve(speech["stem_path"]),
        }
        for mode, path in paths.items():
            wave, rate = sf.read(path, dtype="float32")
            if mode == "clean_speech_upper_bound":
                start = max(0, int((float(speech["onset_seconds"]) - .15) * rate))
                end = min(len(wave), int((float(speech["offset_seconds"]) + .15) * rate))
                wave = wave[start:end]
            audio.append({"array": _asr_audio(wave), "sampling_rate": rate})
            metadata.append({"scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
                             "mode": mode, "reference": speech["transcript"], "audio_path": _portable(path)})
    hypotheses = asr(audio, batch_size=args.asr_batch_size, generate_kwargs={"language": "vi", "task": "transcribe"})
    asr_rows = []
    for meta, hypothesis in zip(metadata, hypotheses):
        text = str(hypothesis["text"]).strip()
        error = _wer(meta["reference"], text)
        asr_rows.append({**meta, "hypothesis": text, "wer_↓": error, "correct_at_wer_0.25": error <= .25})

    summary: dict[str, Any] = {}
    all_modes = (*deployable_modes, "clean_speech_upper_bound")
    for split in ("val", "test"):
        summary[split] = {}
        for mode in all_modes:
            rows = [x for x in asr_rows if x["split"] == split and x["mode"] == mode]
            summary[split][mode] = {
                "scenes": len(rows), "accuracy_at_wer_0.25_↑": float(np.mean([x["correct_at_wer_0.25"] for x in rows])),
                "mean_wer_↓": float(np.mean([x["wer_↓"] for x in rows])),
            }
    selected_mode = min(deployable_modes, key=lambda mode: (summary["val"][mode]["mean_wer_↓"], -summary["val"][mode]["accuracy_at_wer_0.25_↑"]))
    for scene in scenes:
        row = next(x for x in asr_rows if x["scene_id"] == scene["scene_id"] and x["mode"] == selected_mode)
        target = output / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_resolve(row["audio_path"]), target)
    enhanced_path, asr_path = output / "enhancement_items.jsonl", output / "asr_items.jsonl"
    _atomic_text(enhanced_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in enhanced_rows))
    _atomic_text(asr_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in asr_rows))
    receipt = {
        "format": "qces_vietnamese_automotive_deepfilter_receipt_v1", "complete": True,
        "deepfilternet": "DeepFilterNet3", "post_filter": args.post_filter,
        "selection_protocol": "global deployable mode selected by validation mean WER; test transcripts not used",
        "selected_mode": selected_mode, "asr": summary,
        "enhancement_items": _portable(enhanced_path), "asr_items": _portable(asr_path),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected_mode": selected_mode, "asr": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
