#!/usr/bin/env python3
"""Render and score the validation-selected automotive complex-mask model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve
from mixi_understanding.scripts.train_vietnamese_automotive_complexmask import enhance_complex
from mixi_understanding.scripts.train_vietnamese_automotive_tfmask import RATE, TFMaskUNet, _jsonl, _load


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_complexmask_v1/best_sisdr.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_complexmask_eval_v1")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model = TFMaskUNet(input_channels=3, output_channels=2, activation="tanh2").to(device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl"); rows = []
    with torch.inference_mode():
        for scene in scenes:
            mixture = _load(_resolve(scene["mixture_path"])).to(device)
            speech = next(event for event in scene["events"] if event["label"] == "Speech")
            target = _load(_resolve(speech["stem_path"])).to(device)
            length = min(len(mixture), len(target)); mixture, target = mixture[:length], target[:length]
            prediction = enhance_complex(model, mixture[None])[0]
            si, mix_si = float(scale_invariant_sdr(prediction[None], target[None])[0]), float(scale_invariant_sdr(mixture[None], target[None])[0])
            sd, mix_sd = float(scale_dependent_sdr(prediction[None], target[None])[0]), float(scale_dependent_sdr(mixture[None], target[None])[0])
            path = output / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac"
            path.parent.mkdir(parents=True, exist_ok=True); sf.write(path, prediction.cpu().numpy(), RATE, format="FLAC", subtype="PCM_16")
            rows.append({"scene_id": scene["scene_id"], "split": scene["split"], "audio_path": _portable(path),
                         "si_sdr_db_↑": si, "si_sdri_db_↑": si - mix_si,
                         "sd_sdr_db_↑": sd, "sd_sdri_db_↑": sd - mix_sd})
            print(json.dumps({"scene": scene["scene_id"], "si_sdri": si - mix_si}), flush=True)
    summary = {}
    for split in ("val", "test"):
        chosen = [row for row in rows if row["split"] == split]
        summary[split] = {"scenes": len(chosen),
                          "mean_si_sdr_db_↑": float(np.mean([row["si_sdr_db_↑"] for row in chosen])),
                          "mean_si_sdri_db_↑": float(np.mean([row["si_sdri_db_↑"] for row in chosen])),
                          "mean_sd_sdri_db_↑": float(np.mean([row["sd_sdri_db_↑"] for row in chosen]))}
    item_path = output / "items.jsonl"; _atomic_text(item_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    receipt = {"format": "qces_vietnamese_automotive_complexmask_eval_receipt_v1", "complete": True,
               "checkpoint": _portable(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint["epoch"],
               "summary": summary, "items": _portable(item_path)}
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
