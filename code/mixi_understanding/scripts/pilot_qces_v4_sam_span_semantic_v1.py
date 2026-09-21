#!/usr/bin/env python3
"""Oracle-span SAM-Audio architecture screen on deterministic V4 events."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch
import torchaudio.functional as AF

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    load_model, strong_logits_from_waveform,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json, _atomic_torch, _sha256_file,
)


FORMAT = "qces_v4_sam_oracle_span_semantic_pilot_v1"
FRAGMENT_FORMAT = "qces_v4_sam_oracle_span_fragment_v1"


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4"
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=data / "detector_scene_manifest_overlap_dev.jsonl")
    parser.add_argument("--slot-cache", type=Path, default=base / "v4_slot_semantic_head_v1/frozen_slots_dev.pt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--pretrainedsed-checkpoint-name", default="BEATs_strong_1")
    parser.add_argument("--sam-root", type=Path, default=PROJECT_ROOT / "code/baseline/sam-audio")
    parser.add_argument("--sam-checkpoint", type=Path, default=PROJECT_ROOT / "code/baseline/sam-audio/checkpoint/sam-audio-small")
    parser.add_argument("--output-dir", type=Path, default=base / "v4_sam_oracle_span_semantic_pilot_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8211)
    parser.add_argument("--render-first", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(rows, key=lambda row: hashlib.sha256(str(row["scene_id"]).encode()).hexdigest())


def load_audio(path: Path, target_rate: int) -> torch.Tensor:
    waveform, rate = sf.read(path, dtype="float32", always_2d=False)
    value = torch.as_tensor(waveform).float()
    if value.ndim > 1:
        value = value.mean(dim=-1)
    if int(rate) != target_rate:
        value = AF.resample(value, int(rate), target_rate)
    return value


def audio_rank(logits: torch.Tensor, event: Mapping[str, Any], label_id: int) -> int:
    start = max(0, min(249, int(float(event["onset_seconds"]) * 25)))
    end = max(start + 1, min(250, int(math.ceil(float(event["offset_seconds"]) * 25))))
    score = logits[start:end].mean(dim=0)
    order = score.argsort(descending=True)
    return int(torch.where(order == int(label_id))[0].item()) + 1


def rank_summary(ranks: list[int]) -> dict[str, Any]:
    return {
        "events": len(ranks),
        "top1_accuracy_\u2191": sum(rank <= 1 for rank in ranks) / max(len(ranks), 1),
        "top5_accuracy_\u2191": sum(rank <= 5 for rank in ranks) / max(len(ranks), 1),
        "top20_accuracy_\u2191": sum(rank <= 20 for rank in ranks) / max(len(ranks), 1),
        "mean_rank_\u2193": sum(ranks) / max(len(ranks), 1),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fragment_dir = output_dir / "fragments"
    fragment_dir.mkdir(exist_ok=True)
    if args.overwrite and (output_dir / "receipt.json").exists():
        (output_dir / "receipt.json").unlink()

    rows = read_manifest(args.manifest.resolve())
    selected: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for row in rows:
        for event_index, event in enumerate(row["events"]):
            selected.append((row, event, event_index))
            if len(selected) >= args.max_events:
                break
        if len(selected) >= args.max_events:
            break
    cohort = [str(event["event_id"]) for _, event, _ in selected]
    print(json.dumps({"cohort_frozen": True, "events": len(cohort), "ode_steps": args.ode_steps}), flush=True)

    pending = [item for item in selected if not (fragment_dir / f"{hashlib.sha256(str(item[1]['event_id']).encode()).hexdigest()}.pt").exists()]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if pending:
        sam_root = args.sam_root.resolve()
        if str(sam_root) not in sys.path:
            sys.path.insert(0, str(sam_root))
        from sam_audio import SAMAudio, SAMAudioProcessor
        processor = SAMAudioProcessor.from_pretrained(str(args.sam_checkpoint.resolve()))
        sample_rate = int(processor.audio_sampling_rate)
        model = SAMAudio.from_pretrained(
            str(args.sam_checkpoint.resolve()),
            vision_encoder={"dim": 1024, "batch_size": 0, "name": "none"},
            visual_ranker=None, text_ranker=None, span_predictor=None,
        ).eval().half().to(device)
        started = time.time()
        current_scene = None
        mixture48 = None
        for position, (row, event, event_index) in enumerate(pending, start=1):
            if current_scene != row["scene_id"]:
                mixture48 = load_audio(Path(row["mixture_path"]), sample_rate)
                current_scene = row["scene_id"]
            target_clip48 = load_audio(Path(event["component_path"]), sample_rate)
            mixture = mixture48
            length = mixture.numel()
            target = torch.zeros_like(mixture)
            target_start = max(0, min(length, int(round(float(event["onset_seconds"]) * sample_rate))))
            target_end = min(length, target_start + target_clip48.numel())
            target[target_start:target_end] = target_clip48[: target_end - target_start]
            batch = processor(
                audios=[mixture[None]], descriptions=[""],
                anchors=[[["+", float(event["onset_seconds"]), float(event["offset_seconds"])]]],
            ).to(device)
            torch.manual_seed(args.seed + len(cohort) * event_index + position)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                result = model.separate(
                    batch, predict_spans=False, reranking_candidates=1,
                    ode_opt={"method": "midpoint", "options": {"step_size": 1 / args.ode_steps}},
                )
            prediction = result.target[0].float().cpu()[:length]
            residual = result.residual[0].float().cpu()[:length]
            if prediction.numel() != length or residual.numel() != length:
                raise RuntimeError("SAM output length differs from full-scene mixture")
            pred_si = float(scale_invariant_sdr(prediction[None], target[None])[0])
            mix_si = float(scale_invariant_sdr(mixture[None], target[None])[0])
            consistency = float((mixture - prediction - residual).abs().mean())
            fragment = {
                "format": FRAGMENT_FORMAT, "scene_id": row["scene_id"],
                "event_id": event["event_id"], "event_index": event_index,
                "label": event["label"], "label_id": int(event["label_id"]),
                "onset_seconds": float(event["onset_seconds"]),
                "offset_seconds": float(event["offset_seconds"]),
                "prediction_48k": prediction.half(),
                "si_sdr_db": pred_si, "mixture_si_sdr_db": mix_si,
                "si_sdri_db": pred_si - mix_si,
                "native_mixture_consistency_l1": consistency,
            }
            name = hashlib.sha256(str(event["event_id"]).encode()).hexdigest()
            _atomic_torch(fragment, fragment_dir / f"{name}.pt")
            elapsed = time.time() - started
            print(f"event={position}/{len(pending)} label={event['label']} sdri={pred_si-mix_si:.2f} eta={(len(pending)-position)*elapsed/max(position,1)/60:.1f}_min", flush=True)
        del model, processor, batch, result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fragments = []
    for _, event, _ in selected:
        name = hashlib.sha256(str(event["event_id"]).encode()).hexdigest()
        fragments.append(torch.load(fragment_dir / f"{name}.pt", map_location="cpu", weights_only=False))

    detector_payload = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=False)
    labels = detector_payload["labels"]
    detector = load_model(len(labels), args.pretrainedsed_checkpoint_name, device, unfreeze_last_blocks=0)
    detector.load_state_dict(detector_payload["model_state_dict"], strict=True)
    detector.eval().requires_grad_(False)
    sam_ranks: list[int] = []
    for start in range(0, len(fragments), 4):
        batch_fragments = fragments[start : start + 4]
        waveforms = torch.stack([
            AF.resample(fragment["prediction_48k"].float(), 48_000, 16_000)[:160_000]
            for fragment in batch_fragments
        ]).to(device)
        with torch.inference_mode():
            logits = strong_logits_from_waveform(detector, waveforms).cpu()
        for fragment, row_logits in zip(batch_fragments, logits):
            event = {
                "onset_seconds": fragment["onset_seconds"],
                "offset_seconds": fragment["offset_seconds"],
            }
            sam_ranks.append(audio_rank(row_logits, event, int(fragment["label_id"])))
    del detector, detector_payload

    slot_cache = torch.load(args.slot_cache.resolve(), map_location="cpu", weights_only=False)
    cache_index = {str(scene_id): index for index, scene_id in enumerate(slot_cache["scene_id"])}
    baseline_ranks = []
    for row, event, event_index in selected:
        baseline_ranks.append(int(slot_cache["oracle_gold_span_ranks"][cache_index[row["scene_id"]]][event_index]))
    sdri = [float(fragment["si_sdri_db"]) for fragment in fragments]
    consistency = [float(fragment["native_mixture_consistency_l1"]) for fragment in fragments]
    baseline = rank_summary(baseline_ranks)
    sam = rank_summary(sam_ranks)
    separation = {
        "si_sdri_mean_db_\u2191": sum(sdri) / len(sdri),
        "si_sdri_median_db_\u2191": statistics.median(sdri),
        "si_sdri_positive_fraction_\u2191": sum(value > 0 for value in sdri) / len(sdri),
        "native_mixture_consistency_l1_mean_\u2193": sum(consistency) / len(consistency),
        "native_mixture_consistency_l1_max_\u2193": max(consistency),
    }
    gates = {
        "sam_semantic_top1_improves_mixture_by_ge_0_15": sam["top1_accuracy_\u2191"] >= baseline["top1_accuracy_\u2191"] + 0.15,
        "sam_median_si_sdri_ge_1dB": separation["si_sdri_median_db_\u2191"] >= 1.0,
    }
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    for index, fragment in enumerate(fragments[: args.render_first]):
        sf.write(audio_dir / f"{index:02d}_{fragment['label']}.flac", fragment["prediction_48k"].float().numpy(), 48_000)
    receipt = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "paper_eligible": False,
        "access": "oracle_span_architecture_upper_bound",
        "cohort_selection": "scene SHA256 order then manifest event order; frozen before inference",
        "cohort_event_ids": cohort, "qa_question_or_answer_used_as_input": False,
        "baseline_mixture_oracle_span_semantic": baseline,
        "sam_oracle_span_semantic": sam, "separation": separation,
        "success_gates": gates,
        "decision": "use_sam_temporal_frontend_and_distill" if all(gates.values()) else "sam_span_insufficient_train_pit_separator",
        "inputs": {
            "manifest_sha256": _sha256_file(args.manifest.resolve()),
            "slot_cache_sha256": _sha256_file(args.slot_cache.resolve()),
            "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve()),
            "sam_checkpoint_sha256": _sha256_file(args.sam_checkpoint.resolve() / "checkpoint.pt"),
        },
        "items": [{key: value for key, value in fragment.items() if key != "prediction_48k"} for fragment in fragments],
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "baseline": baseline, "sam": sam, "separation": separation, "gates": gates, "decision": receipt["decision"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
