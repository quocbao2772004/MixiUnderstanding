#!/usr/bin/env python3
"""Factorize SAM-Audio span-only extraction on overlapping answer events.

The proposed condition uses a positive answer-event interval and, when
available, a negative anchor-only interval where the answer has not started
(or has already ended).  It never uses the answer label.  Oracle answer text
plus the same positive span is reported only as a semantic upper bound.

V1 uses annotated intervals to test the separator capability before predicted
temporal errors are introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_anchor_residualization_v1 import (
    full_component,
    jsonl,
    read_mono,
    resolve_under,
    window_mask,
)
from mixi_understanding.scripts.evaluate_sam_audio_baselines import (
    load_sam_audio,
    sample_seed,
)


FORMAT = "qces_sam_span_contrast_factorization_v1"
SOURCE_RATE = 16_000
MODES = (
    "mixture__oracle_answer_window",
    "sam__oracle_answer_span_only",
    "sam__oracle_answer_span_plus_anchor_negative",
    "sam__oracle_answer_text_plus_span",
)


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full191_overlap_onset_stress_test_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", type=Path, default=data / "scene_manifest_test.jsonl")
    parser.add_argument("--qa-manifest", type=Path, default=data / "qa_manifest_test.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=data)
    parser.add_argument(
        "--model", type=Path,
        default=PROJECT_ROOT / "code/baseline/sam-audio/checkpoint/sam-audio-small",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_sam_span_contrast_overlap_stress_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2203)
    parser.add_argument("--ode-steps", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--render-first", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resample(value: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    value = value.reshape(1, -1)
    return AF.resample(value, source_rate, target_rate)[0] if source_rate != target_rate else value[0]


def metric(candidate: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    length = min(candidate.numel(), target.numel(), baseline.numel())
    candidate, target, baseline = candidate[:length].float(), target[:length].float(), baseline[:length].float()
    sd = float(scale_dependent_sdr(candidate[None], target[None])[0])
    base_sd = float(scale_dependent_sdr(baseline[None], target[None])[0])
    si = float(scale_invariant_sdr(candidate[None], target[None])[0])
    cosine = float(torch.dot(candidate, target) / (candidate.norm() * target.norm()).clamp_min(1e-8))
    target_power = target.square().mean().clamp_min(1e-8)
    return {
        "sd_sdr_db_↑": sd,
        "sd_sdri_vs_mixture_window_db_↑": sd - base_sd,
        "si_sdr_db_↑": si,
        "waveform_cosine_↑": cosine,
        "normalized_mse_↓": float((candidate - target).square().mean() / target_power),
    }


def negative_anchor_only(qa: Mapping[str, Any], anchor: Mapping[str, Any], answer: Mapping[str, Any]) -> list[float | str] | None:
    if str(qa["relation"]) == "after":
        start = float(anchor["onset_seconds"])
        end = min(float(anchor["offset_seconds"]), float(answer["onset_seconds"]))
    else:
        start = max(float(anchor["onset_seconds"]), float(answer["offset_seconds"]))
        end = float(anchor["offset_seconds"])
    return ["-", start, end] if end - start >= 0.04 else None


def run_sam(processor, model, mixture: torch.Tensor, description: str, anchors, device: torch.device, seed: int, ode_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = processor(audios=[mixture.reshape(1, -1)], descriptions=[description], anchors=[anchors]).to(device)
    torch.manual_seed(seed)
    context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else __import__("contextlib").nullcontext()
    with torch.inference_mode(), context:
        result = model.separate(
            batch, predict_spans=False, reranking_candidates=1,
            ode_opt={"method": "midpoint", "options": {"step_size": 1 / ode_steps}},
        )
    return result.target[0].reshape(-1).float().cpu(), result.residual[0].reshape(-1).float().cpu()


def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in MODES:
        rows = [item for item in items if item["mode"] == mode]
        entry = {"records": len(rows)}
        for key in ("sd_sdr_db_↑", "sd_sdri_vs_mixture_window_db_↑", "si_sdr_db_↑", "waveform_cosine_↑", "normalized_mse_↓"):
            values = [float(row["metrics"][key]) for row in rows]
            entry[f"mean_{key}"] = float(np.mean(values))
            entry[f"median_{key}"] = float(median(values))
        if mode != MODES[0]:
            entry["fraction_improved_sd_sdr_vs_mixture_↑"] = float(np.mean([
                row["metrics"]["sd_sdri_vs_mixture_window_db_↑"] > 0 for row in rows
            ]))
        result[mode] = entry
    return result


def main() -> None:
    args = parse_args()
    if args.ode_steps < 1 or args.max_records < 1 or args.render_first < 0:
        raise SystemExit("ode-steps/max-records must be positive; render-first non-negative")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    root = args.dataset_root.resolve()
    scenes = {str(row["scene_id"]): row for row in jsonl(args.scene_manifest)}
    qas = [row for row in jsonl(args.qa_manifest) if not bool(row["no_evidence"])][:args.max_records]
    device = torch.device(args.device)
    processor, model = load_sam_audio(args.model.resolve(), device)
    sample_rate = int(processor.audio_sampling_rate)
    items = []
    for index, qa in enumerate(qas):
        scene = scenes[str(qa["scene_id"])]
        events = {str(event["event_id"]): event for event in scene["events"]}
        anchor = events[str(qa["anchor_event_id"])]
        answer = events[str(qa["answer_event_id"])]
        mixture_source = torch.from_numpy(read_mono(resolve_under(root, str(scene["mixture_path"]))))
        target_source = full_component(root, answer, mixture_source.numel())
        mask_source = window_mask(mixture_source.numel(), answer)
        mixture = resample(mixture_source, SOURCE_RATE, sample_rate)
        target = resample(target_source, SOURCE_RATE, sample_rate)
        mask = resample(mask_source, SOURCE_RATE, sample_rate).clamp(0, 1)
        baseline = mixture * mask
        positive = ["+", float(answer["onset_seconds"]), float(answer["offset_seconds"])]
        negative = negative_anchor_only(qa, anchor, answer)
        anchors_contrast = [positive] + ([] if negative is None else [negative])
        paired_seed = sample_seed(args.seed, str(qa["item_id"]))
        span_only, _ = run_sam(processor, model, mixture, "", [positive], device, paired_seed, args.ode_steps)
        contrast, _ = run_sam(processor, model, mixture, "", anchors_contrast, device, paired_seed, args.ode_steps)
        answer_text = str(answer["label"]).replace("_", " ").lower()
        text_span, _ = run_sam(processor, model, mixture, answer_text, [positive], device, paired_seed, args.ode_steps)
        candidates = {
            MODES[0]: baseline,
            MODES[1]: span_only,
            MODES[2]: contrast,
            MODES[3]: text_span,
        }
        for mode, candidate in candidates.items():
            values = metric(candidate, target, baseline)
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"non-finite metric {qa['item_id']} {mode}")
            items.append({
                "item_id": qa["item_id"], "scene_id": qa["scene_id"], "question": qa["question"],
                "anchor_label": anchor["label"], "answer_label_diagnostic_only": answer["label"],
                "mode": mode, "anchors": ([positive] if mode != MODES[2] else anchors_contrast),
                "negative_anchor_available": negative is not None, "metrics": values,
            })
        if index < args.render_first:
            case = output / "audio" / str(qa["scene_id"])
            case.mkdir(parents=True, exist_ok=True)
            for file_index, (mode, candidate) in enumerate(candidates.items()):
                sf.write(case / f"{file_index:02d}_{mode}.flac", candidate.numpy(), sample_rate, format="FLAC")
            sf.write(case / "04_target_answer.flac", target.numpy(), sample_rate, format="FLAC")
            (case / "metadata.json").write_text(json.dumps({
                "question": qa["question"], "anchor": anchor["label"], "answer": answer["label"],
                "positive": positive, "negative": negative,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"evaluated {index + 1}/{len(qas)}", flush=True)
    items_path = output / "items.jsonl"
    items_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items), encoding="utf-8")
    checkpoint = args.model.resolve() / "checkpoint.pt"
    report = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "protocol": {"answer_label_used_by_proposed_modes": False, "oracle_intervals": True,
                     "oracle_answer_text_is_upper_bound_only": True, "records": len(qas),
                     "ode_steps": args.ode_steps, "reranking_candidates": 1, "paired_seed_across_modes": True},
        "summary": summarize(items),
        "model": {"path": str(args.model.resolve()), "checkpoint_sha256": hash_file(checkpoint)},
        "artifacts": {"items": str(items_path), "items_sha256": hash_file(items_path)},
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
