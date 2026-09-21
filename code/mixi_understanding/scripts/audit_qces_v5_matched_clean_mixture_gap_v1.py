#!/usr/bin/env python3
"""Paired clean-component vs mixture oracle-span ATST semantic audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import models.prediction_wrapper as prediction_wrapper_module
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper
from mixi_understanding.scripts.audit_qces_v5_targeted_logit_calibration_v1 import compact_metric
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _atomic_torch, _sha256_file
from mixi_understanding.scripts.train_qces_v5_atst_targeted_enrichment_v1 import load_head


FORMAT = "qces_v5_matched_clean_mixture_gap_v1"
SAMPLE_RATE = 16_000
NUM_SAMPLES = 160_000
NUM_FRAMES = 250
HOP_SAMPLES = 640


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/var/tmp/qces_v5_matched_eval_v1/detector_scene_manifest_matched_eval.jsonl"))
    parser.add_argument("--mixture-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_matched_eval_v1/atst_features_matched_eval.pt"))
    parser.add_argument("--checkpoint", type=Path, default=base / "v5_atst_oracle_semantic_screen_v1/atst_oracle_semantic_head_v1_best.pt")
    parser.add_argument("--clean-logit-cache", type=Path, default=Path("/var/tmp/qces_v5_matched_clean_mixture_gap_v1/clean_logits.pt"))
    parser.add_argument("--output", type=Path, default=base / "v5_atst_targeted_enrichment_v1/matched_clean_mixture_gap_v1.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7411)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_scenes(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]


def event_rows(scenes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes):
        for event in scene["events"]:
            result.append({"scene_index": scene_index, "scene_id": scene["scene_id"], **event})
    return result


def load_component_canvas(event: dict[str, Any]) -> torch.Tensor:
    audio, sample_rate = sf.read(str(event["component_path"]), dtype="float32", always_2d=True)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"unexpected component sample rate: {sample_rate}")
    mono = torch.from_numpy(audio.mean(axis=1))
    canvas = torch.zeros(NUM_SAMPLES, dtype=torch.float32)
    begin = int(event.get("onset_frame", round(float(event["onset_seconds"]) / 0.04))) * HOP_SAMPLES
    usable = min(mono.numel(), NUM_SAMPLES - begin)
    canvas[begin : begin + usable] = mono[:usable]
    return canvas


def pad_spans(features: torch.Tensor, batch_events: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    spans: list[torch.Tensor] = []
    for index, event in enumerate(batch_events):
        start = max(0, min(NUM_FRAMES - 1, int(event.get("onset_frame", 0))))
        end = max(start + 1, min(NUM_FRAMES, int(event.get("offset_frame", start + 1))))
        spans.append(features[index, start:end])
    maximum = max(span.shape[0] for span in spans)
    padded = features.new_zeros(len(spans), maximum, features.shape[-1])
    mask = torch.zeros(len(spans), maximum, dtype=torch.bool, device=features.device)
    for index, span in enumerate(spans):
        padded[index, : span.shape[0]] = span
        mask[index, : span.shape[0]] = True
    return padded, mask


@torch.inference_mode()
def clean_logits(
    backbone: PredictionsWrapper,
    head: torch.nn.Module,
    events: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for begin in range(0, len(events), batch_size):
        batch_events = events[begin : begin + batch_size]
        waveforms = torch.stack([load_component_canvas(event) for event in batch_events]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            mel = backbone.mel_forward(waveforms)
            features = backbone.model(mel)
        if int(features.shape[1]) != NUM_FRAMES:
            features = F.interpolate(
                features.transpose(1, 2), size=NUM_FRAMES, mode="linear", align_corners=False
            ).transpose(1, 2)
        spans, mask = pad_spans(features.float(), batch_events)
        output.append(head(spans, mask).cpu())
        processed = min(begin + len(batch_events), len(events))
        if processed == len(batch_events) or processed % 256 < len(batch_events):
            print(f"clean_component_inference {processed}/{len(events)}", flush=True)
    return torch.cat(output)


def bin_metrics(
    mix_scores: torch.Tensor,
    clean_scores: torch.Tensor,
    targets: torch.Tensor,
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    bins: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        overlap = float(event.get("overlap_fraction", 0.0))
        sir = float(event.get("active_sir_db", 40.0))
        duration = float(event["offset_seconds"]) - float(event["onset_seconds"])
        bins["overlap:zero" if overlap <= 1e-9 else "overlap:(0,.25]" if overlap <= 0.25 else "overlap:(.25,.75]" if overlap <= 0.75 else "overlap:(.75,1]"].append(index)
        bins["sir:<0" if sir < 0 else "sir:[0,5)" if sir < 5 else "sir:[5,10)" if sir < 10 else "sir:>=10"].append(index)
        bins["duration:<=0.5" if duration <= 0.5 else "duration:(0.5,1.5]" if duration <= 1.5 else "duration:>1.5"].append(index)
    result: dict[str, Any] = {}
    for name, indices in sorted(bins.items()):
        selected = torch.tensor(indices, dtype=torch.long)
        mix_correct = mix_scores[selected].argmax(1).eq(targets[selected])
        clean_correct = clean_scores[selected].argmax(1).eq(targets[selected])
        result[name] = {
            "events": len(indices),
            "mixture_top1_↑": float(mix_correct.float().mean()),
            "clean_top1_↑": float(clean_correct.float().mean()),
            "clean_minus_mixture": float(clean_correct.float().mean() - mix_correct.float().mean()),
            "either_oracle_top1_↑": float((mix_correct | clean_correct).float().mean()),
        }
    return result


def scene_bootstrap(
    mix_correct: np.ndarray,
    clean_correct: np.ndarray,
    scene_index: np.ndarray,
    *,
    num_scenes: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, num_scenes, size=num_scenes)
        weights = np.bincount(sample, minlength=num_scenes)[scene_index]
        delta[index] = ((clean_correct - mix_correct) * weights).sum() / max(weights.sum(), 1)
    return {
        "replicates": replicates,
        "mean_delta": float(delta.mean()),
        "ci95_lower": float(np.quantile(delta, 0.025)),
        "ci95_upper": float(np.quantile(delta, 0.975)),
        "probability_delta_gt_0": float((delta > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    scenes = read_scenes(args.manifest.resolve())
    events = event_rows(scenes)
    mix_cache = torch.load(args.mixture_cache.resolve(), map_location="cpu", weights_only=False)
    targets = torch.cat(list(mix_cache["labels"]))
    if len(events) != int(targets.numel()):
        raise RuntimeError("manifest/cache event count mismatch")
    manifest_targets = torch.tensor([int(event["label_id"]) for event in events])
    if not torch.equal(targets, manifest_targets):
        raise RuntimeError("manifest/cache event order mismatch")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    head, payload = load_head(args.checkpoint.resolve(), device)
    # Obtain mixture logits from the already-exported cache with the exact same head.
    from mixi_understanding.scripts.train_qces_v5_atst_targeted_enrichment_v1 import predictions
    from mixi_understanding.scripts.train_qces_v5_atst_oracle_semantic_screen_v1 import CachedSceneDataset, cached_collate
    from torch.utils.data import DataLoader
    loader = DataLoader(CachedSceneDataset(mix_cache), batch_size=32, shuffle=False, num_workers=0, collate_fn=cached_collate)
    mix_scores, repeated_targets = predictions(head, loader, device)
    if not torch.equal(targets, repeated_targets):
        raise RuntimeError("mixture inference target order mismatch")

    prediction_wrapper_module.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    backbone = PredictionsWrapper(
        ATSTWrapper(), checkpoint="ATST-F_strong_1",
        n_classes_strong=len(payload["labels"]), n_classes_weak=len(payload["labels"]),
        seq_model_type=None, head_type="linear",
    ).to(device)
    backbone.eval().requires_grad_(False)
    clean_scores = clean_logits(
        backbone, head, events, device=device, batch_size=args.batch_size, amp=args.amp
    )
    cache_path = args.clean_logit_cache.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch({
        "format": FORMAT + "_clean_logits", "scene_id": [event["scene_id"] for event in events],
        "event_id": [event["event_id"] for event in events], "targets": targets,
        "logits": clean_scores,
    }, cache_path)

    mix_correct = mix_scores.argmax(1).eq(targets)
    clean_correct = clean_scores.argmax(1).eq(targets)
    result = {
        "mixture_oracle_span": compact_metric(mix_scores, targets),
        "clean_component_same_span": compact_metric(clean_scores, targets),
        "paired": {
            "both_correct": int((mix_correct & clean_correct).sum()),
            "mixture_only_correct": int((mix_correct & ~clean_correct).sum()),
            "clean_only_correct": int((~mix_correct & clean_correct).sum()),
            "both_wrong": int((~mix_correct & ~clean_correct).sum()),
            "either_oracle_top1_↑": float((mix_correct | clean_correct).float().mean()),
        },
        "bins": bin_metrics(mix_scores, clean_scores, targets, events),
    }
    scene_index = np.asarray([int(event["scene_index"]) for event in events], dtype=np.int64)
    bootstrap = scene_bootstrap(
        mix_correct.numpy().astype(np.float64), clean_correct.numpy().astype(np.float64),
        scene_index, num_scenes=len(scenes), replicates=args.bootstrap_replicates, seed=args.seed,
    )
    clean_top1 = result["clean_component_same_span"]["top1_accuracy_↑"]
    mix_top1 = result["mixture_oracle_span"]["top1_accuracy_↑"]
    report = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "paired official-eval source diagnostic with oracle event intervals",
        "data": {"scenes": len(scenes), "events": len(events), "classes": len(payload["labels"])},
        "metrics": result,
        "clean_minus_mixture_top1": clean_top1 - mix_top1,
        "scene_bootstrap_clean_minus_mixture": bootstrap,
        "decision": (
            "overlap_interference_is_primary_semantic_bottleneck"
            if clean_top1 - mix_top1 >= 0.15
            else "clean_source_or_representation_error_is_also_major"
        ),
        "artifacts": {
            "manifest_sha256": _sha256_file(args.manifest.resolve()),
            "mixture_cache_sha256": _sha256_file(args.mixture_cache.resolve()),
            "clean_logit_cache_sha256": _sha256_file(cache_path),
            "checkpoint_sha256": _sha256_file(args.checkpoint.resolve()),
        },
    }
    _atomic_json(report, args.output.resolve())
    print(json.dumps({
        "mixture_top1": mix_top1, "clean_top1": clean_top1,
        "delta": clean_top1 - mix_top1, "paired": result["paired"],
        "bootstrap": bootstrap, "decision": report["decision"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
