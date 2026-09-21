#!/usr/bin/env python3
"""Small SAM-Audio speech extraction smoke test on automotive scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio.functional as AF

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load(path: Path, rate: int) -> torch.Tensor:
    wave, source_rate = sf.read(path, dtype="float32", always_2d=True)
    value = torch.from_numpy(wave.mean(axis=1)).float()
    return AF.resample(value, source_rate, rate) if source_rate != rate else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--sam-audio-root", type=Path, default=PROJECT_ROOT / "code/baseline/sam-audio")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "code/baseline/sam-audio/checkpoint/sam-audio-small")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_sam_audio_smoke_v1")
    parser.add_argument("--prompt", default="speech")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-scenes", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sam_audio_root.resolve()))
    from sam_audio import SAMAudio, SAMAudioProcessor

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    vision = {"dim": 1024, "batch_size": 300, "name": "none", "normalize_feature": True,
              "interpolation_mode": "BICUBIC", "image_size": 336}
    model = SAMAudio.from_pretrained(
        str(args.checkpoint.resolve()), vision_encoder=vision, visual_ranker=None,
        text_ranker=None, span_predictor=None,
    ).eval().cuda()
    processor = SAMAudioProcessor.from_pretrained(str(args.checkpoint.resolve()))
    sample_rate = int(processor.audio_sampling_rate)
    all_scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")
    scenes = all_scenes if args.split == "all" else [x for x in all_scenes if x["split"] == args.split]
    scenes = scenes[:args.max_scenes] if args.max_scenes else scenes
    rows = []
    for scene in scenes:
        batch = processor(audios=[str(_resolve(scene["mixture_path"]))], descriptions=[args.prompt]).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            result = model.separate(batch, predict_spans=False, reranking_candidates=1)
        prediction = result.target[0].float().cpu()
        residual = result.residual[0].float().cpu()
        target_event = next(x for x in scene["events"] if x["label"] == "Speech")
        target = _load(_resolve(target_event["stem_path"]), sample_rate)
        mixture = _load(_resolve(scene["mixture_path"]), sample_rate)
        length = min(len(prediction), len(target), len(mixture))
        pred_si = float(scale_invariant_sdr(prediction[:length][None], target[:length][None])[0])
        mix_si = float(scale_invariant_sdr(mixture[:length][None], target[:length][None])[0])
        target_path = output / "audio" / f"{scene['scene_id']}_target.flac"
        residual_path = output / "audio" / f"{scene['scene_id']}_residual.flac"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target_path, prediction.numpy(), sample_rate, format="FLAC", subtype="PCM_16")
        sf.write(residual_path, residual.numpy(), sample_rate, format="FLAC", subtype="PCM_16")
        row = {"scene_id": scene["scene_id"], "prompt": args.prompt, "target_path": _portable(target_path),
               "residual_path": _portable(residual_path), "si_sdr_db_↑": pred_si,
               "mixture_si_sdr_db_↑": mix_si, "si_sdri_db_↑": pred_si - mix_si}
        rows.append(row)
        print(json.dumps(row), flush=True)
    _atomic_text(output / "items.jsonl", "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
