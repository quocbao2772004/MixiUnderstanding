#!/usr/bin/env python3
"""Audit whether joint separation improves frozen semantic recognition.

Gold PIT association and oracle boxes are diagnostic upper-bound tools only;
neither QA questions nor answers are inputs to the semantic classifier.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from mixi_understanding.scripts.train_qces_long_short_semantic_teacher_v2 import (
    LongHeadConfig,
    LongSemanticHead,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    interpolate_sequence,
    load_model,
    load_ontology,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _sha256_file
from mixi_understanding.scripts.train_qces_v5_joint_convtasnet_micro_v1 import (
    SAMPLE_RATE,
    SAMPLES,
    SceneDataset,
    match_active,
    model_forward,
)
from mixi_understanding.scripts.train_qces_v5_joint_convtasnet_pilot_v1 import stable_scenes


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "qces_v5_joint_separator_semantic_audit_v1"
NUM_FRAMES = 250


def parse_args() -> argparse.Namespace:
    data = Path("/var/tmp/qces_full188_tiered_realistic_v5")
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=data / "detector_scene_manifest_tiered_dev.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--separator-checkpoint", type=Path, default=base / "v5_joint_convtasnet_pilot_v1/joint_convtasnet_pilot_v1_best.pt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--long-teacher-checkpoint", type=Path, default=base / "long_short_semantic_teacher_v2/long_short_semantic_teacher_v2_best.pt")
    parser.add_argument("--output", type=Path, default=base / "v5_joint_convtasnet_pilot_v1/semantic_audit_dev.json")
    parser.add_argument("--pretrained-checkpoint", default="BEATs_strong_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8221)
    parser.add_argument("--max-scenes", type=int, default=128)
    parser.add_argument("--num-sources", type=int, default=6)
    parser.add_argument("--semantic-batch-size", type=int, default=12)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def box_mask(boxes: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(boxes.shape[0], NUM_FRAMES, device=boxes.device)
    for index, (start, end) in enumerate(boxes.tolist()):
        left = max(0, min(NUM_FRAMES, int(math.floor(start / SAMPLES * NUM_FRAMES))))
        right = max(left + 1, min(NUM_FRAMES, int(math.ceil(end / SAMPLES * NUM_FRAMES))))
        result[index, left:right] = 1.0
    return result


def energy_mask(waveforms: torch.Tensor) -> torch.Tensor:
    frame_samples = SAMPLES // NUM_FRAMES
    rms = waveforms.reshape(waveforms.shape[0], NUM_FRAMES, frame_samples).square().mean(dim=-1).sqrt()
    maximum = rms.amax(dim=-1, keepdim=True)
    median = rms.median(dim=-1, keepdim=True).values
    threshold = torch.maximum(maximum * 0.04, median * 3.0)
    active = (rms >= threshold).float()
    # Preserve short transients and fill one-frame holes.
    active = F.max_pool1d(active[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    inactive_holes = 1.0 - active
    active = 1.0 - F.max_pool1d(inactive_holes[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    empty = active.sum(dim=-1) < 1
    if empty.any():
        active[empty, rms[empty].argmax(dim=-1)] = 1.0
    return active


def masked_stats(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    native = F.interpolate(mask[:, None], size=features.shape[1], mode="linear", align_corners=False)[:, 0]
    native = (native >= 0.25).float()
    weight = native / native.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = torch.einsum("bt,btd->bd", weight, features)
    centered = features - mean[:, None]
    variance = torch.einsum("bt,btd->bd", weight, centered.square())
    maximum = features.masked_fill(native[..., None] <= 0, float("-inf")).amax(dim=1)
    return torch.cat((mean, variance.clamp_min(1e-6).sqrt(), maximum), dim=-1)


@torch.inference_mode()
def semantic_logits(
    backbone: torch.nn.Module,
    teacher: LongSemanticHead,
    waveforms: torch.Tensor,
    masks: torch.Tensor,
    *,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    device = next(backbone.parameters()).device
    for start in range(0, waveforms.shape[0], batch_size):
        # Keep the full 458-event audit on CPU and stage only one batch on the
        # shared T4.  Qwen/live-demo services remain resident during this run.
        audio = waveforms[start : start + batch_size].to(device, non_blocking=True)
        mask = masks[start : start + batch_size].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(amp and audio.is_cuda)):
            mel = backbone.mel_forward(audio)
            features = backbone.model(mel).float()
            dense_features = interpolate_sequence(features, backbone.seq_len)
            dense_logits = backbone.strong_head(backbone.seq_model(dense_features)).float()
            stats = masked_stats(features, mask)
            weight = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pooled_r1 = torch.einsum("bt,btc->bc", weight, dense_logits)
            logits, _ = teacher(stats, pooled_r1)
        rows.append(logits.float().cpu())
    return torch.cat(rows)


def metric(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    top = logits.topk(5, dim=-1).indices
    correct = top[:, 0].eq(labels)
    class_total: Counter[int] = Counter(labels.tolist())
    class_correct: Counter[int] = Counter(label for label, ok in zip(labels.tolist(), correct.tolist()) if ok)
    return {
        "events": int(labels.numel()),
        "top1_accuracy_\u2191": float(correct.float().mean()),
        "top5_accuracy_\u2191": float((top == labels[:, None]).any(dim=-1).float().mean()),
        "macro_top1_accuracy_\u2191": float(sum(class_correct[key] / value for key, value in class_total.items()) / len(class_total)),
        "observed_classes": len(class_total),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    labels = load_ontology(args.ontology.resolve())
    label_to_id = {label: index for index, label in enumerate(labels)}
    scenes = stable_scenes(args.manifest.resolve(), args.max_scenes, args.num_sources, args.seed + 1)
    dataset = SceneDataset(scenes, args.num_sources)
    device = make_device(args.device)
    separator_payload = torch.load(args.separator_checkpoint.resolve(), map_location="cpu", weights_only=True)
    architecture = separator_payload["architecture"]
    separator = torchaudio.models.ConvTasNet(
        num_sources=architecture["num_sources"],
        enc_num_feats=architecture["enc_num_feats"],
        msk_num_feats=architecture["mask_num_feats"],
        msk_num_hidden_feats=architecture.get("mask_num_hidden_feats", architecture.get("mask_hidden_feats")),
        msk_num_layers=architecture["mask_num_layers"],
        msk_num_stacks=architecture["mask_num_stacks"],
    )
    separator.load_state_dict(separator_payload["model_state_dict"], strict=True)
    separator.to(device).eval()
    detector_payload = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if list(detector_payload.get("labels") or []) != labels:
        raise ValueError("detector ontology mismatch")
    backbone = load_model(len(labels), args.pretrained_checkpoint, device)
    backbone.load_state_dict(detector_payload["model_state_dict"], strict=True)
    backbone.eval().requires_grad_(False)
    teacher_payload = torch.load(args.long_teacher_checkpoint.resolve(), map_location="cpu", weights_only=True)
    teacher = LongSemanticHead(LongHeadConfig(**teacher_payload["long_config"]))
    teacher.load_state_dict(teacher_payload["long_model_state_dict"], strict=True)
    teacher.to(device).eval()

    clean_audio: list[torch.Tensor] = []
    raw_audio: list[torch.Tensor] = []
    separated_audio: list[torch.Tensor] = []
    oracle_masks: list[torch.Tensor] = []
    gold_labels: list[int] = []
    overlap_fraction: list[float] = []
    processed = 0
    for sample in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        mixture = sample["mixture"].to(device)
        targets = sample["targets"][0].to(device)
        boxes = sample["boxes"][0].to(device)
        count = int(sample["num_active"].item())
        prediction = model_forward(separator, mixture, args.amp)[0]
        output_indices, target_indices = match_active(prediction, targets[:count])
        for output_index, target_index in zip(output_indices.tolist(), target_indices.tolist()):
            start, end = boxes[target_index].tolist()
            raw = torch.zeros_like(mixture[0])
            raw[start:end] = mixture[0, start:end]
            clean_audio.append(targets[target_index].cpu())
            raw_audio.append(raw.cpu())
            separated_audio.append(prediction[output_index].cpu())
            oracle_masks.append(box_mask(boxes[target_index : target_index + 1])[0].cpu())
            gold_labels.append(label_to_id[sample["labels"][target_index][0]])
            overlap_fraction.append(float(sample["overlap"][0, target_index]))
        processed += 1
        if processed == 1 or processed % 32 == 0:
            print(f"separation {processed}/{len(dataset)}", flush=True)
    clean = torch.stack(clean_audio)
    raw = torch.stack(raw_audio)
    separated = torch.stack(separated_audio)
    masks = torch.stack(oracle_masks)
    energy_masks = energy_mask(separated)
    logits = {
        "clean_target_oracle_box": semantic_logits(backbone, teacher, clean, masks, batch_size=args.semantic_batch_size, amp=args.amp),
        "raw_mixture_oracle_box": semantic_logits(backbone, teacher, raw, masks, batch_size=args.semantic_batch_size, amp=args.amp),
        "separated_oracle_box": semantic_logits(backbone, teacher, separated, masks, batch_size=args.semantic_batch_size, amp=args.amp),
        "separated_energy_mask": semantic_logits(backbone, teacher, separated, energy_masks, batch_size=args.semantic_batch_size, amp=args.amp),
    }
    target = torch.tensor(gold_labels, dtype=torch.long)
    metrics = {name: metric(value, target) for name, value in logits.items()}
    raw_correct = logits["raw_mixture_oracle_box"].argmax(dim=-1).eq(target)
    separated_correct = logits["separated_oracle_box"].argmax(dim=-1).eq(target)
    transitions = {
        "raw_wrong_to_separated_right_\u2191": int((~raw_correct & separated_correct).sum()),
        "raw_right_to_separated_wrong_\u2193": int((raw_correct & ~separated_correct).sum()),
        "both_right": int((raw_correct & separated_correct).sum()),
        "both_wrong": int((~raw_correct & ~separated_correct).sum()),
        "net_correct_gain": int(separated_correct.sum() - raw_correct.sum()),
    }
    overlap_tensor = torch.tensor(overlap_fraction)
    overlap_metrics = {}
    for name, selected in {
        "any_overlap_ge_0_05": overlap_tensor >= 0.05,
        "heavy_overlap_ge_0_25": overlap_tensor >= 0.25,
    }.items():
        overlap_metrics[name] = {
            mode: metric(value[selected], target[selected])
            for mode, value in logits.items()
        } if selected.any() else None
    gates = {
        "separated_oracle_box_top1_improves_raw": metrics["separated_oracle_box"]["top1_accuracy_\u2191"] > metrics["raw_mixture_oracle_box"]["top1_accuracy_\u2191"],
        "net_correct_gain_positive": transitions["net_correct_gain"] > 0,
        "energy_mask_retains_at_least_95pct_of_oracle_box_top1": metrics["separated_energy_mask"]["top1_accuracy_\u2191"] >= 0.95 * metrics["separated_oracle_box"]["top1_accuracy_\u2191"],
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "diagnostic source-disjoint dev audit; gold PIT association and oracle boxes are not deployable inputs",
        "scenes": len(dataset),
        "events": len(gold_labels),
        "classes": len(labels),
        "metrics": metrics,
        "overlap_metrics": overlap_metrics,
        "transitions": transitions,
        "gates": gates,
        "decision": "integrate_separator_with_event_graph" if all(gates.values()) else "separator_does_not_fix_semantics",
        "artifacts": {
            "separator_checkpoint": str(args.separator_checkpoint.resolve()),
            "separator_checkpoint_sha256": _sha256_file(args.separator_checkpoint.resolve()),
            "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve()),
            "long_teacher_checkpoint_sha256": _sha256_file(args.long_teacher_checkpoint.resolve()),
        },
    }
    _atomic_json(receipt, args.output.resolve())
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
