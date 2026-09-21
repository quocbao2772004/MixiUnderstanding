#!/usr/bin/env python3
"""Separate human speech from one uploaded car-noise audio file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _portable


def _load(path: Path, sample_rate: int):
    import torch
    import torchaudio.functional as AF

    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    tensor = torch.from_numpy(waveform.mean(axis=1)).float()
    if rate != sample_rate:
        tensor = AF.resample(tensor, rate, sample_rate)
    return tensor


def _run_frcrn(input_path: Path, output_path: Path) -> None:
    import torch
    from clearvoice import ClearVoice

    source = _load(input_path, 16_000)
    enhancer = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    with torch.inference_mode():
        enhanced = enhancer(source[None].numpy())
    output = np.asarray(enhanced)[0].astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, output, 16_000, format="FLAC", subtype="PCM_16")


def _run_audiosep(input_path: Path, output_path: Path, prompt: str) -> None:
    import torch
    import torchaudio.functional as AF
    from mixi_understanding.scripts.evaluate_audiosep_baselines import _load_separator, encode_prompts

    sample_rate = 32_000
    source = _load(input_path, sample_rate)
    args = SimpleNamespace(
        audiosep_root=PROJECT_ROOT / "code/baseline/audiosep",
        audiosep_config=PROJECT_ROOT / "code/baseline/audiosep/config/audiosep_base.yaml",
        audiosep_checkpoint=PROJECT_ROOT / "code/baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = encode_prompts(args.audiosep_root, args.audiosep_checkpoint, [prompt], batch_size=1)[prompt][None].to(device)
    separator = _load_separator(args, device)
    with torch.inference_mode():
        output = separator({"mixture": source[None, None].to(device), "condition": embedding})["waveform"][0, 0].detach().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, output.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("frcrn", "audiosep"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="a person speaking")
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.method == "frcrn":
        _run_frcrn(args.input.resolve(), args.output.resolve())
    else:
        _run_audiosep(args.input.resolve(), args.output.resolve(), args.prompt)
    receipt = {
        "format": "qces_uploaded_car_audio_inference_v1",
        "complete": True,
        "method": args.method,
        "prompt": args.prompt if args.method == "audiosep" else None,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
