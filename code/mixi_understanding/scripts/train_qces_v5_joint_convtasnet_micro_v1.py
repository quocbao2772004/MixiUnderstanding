#!/usr/bin/env python3
"""Micro-overfit a joint scene separator before spending on a full run.

This is deliberately independent from the old event-wise temporal separator.
One mixture is mapped to all scene sources in one pass and permutation
invariant training (PIT) matches outputs to the exact V5 component stems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset


FORMAT = "qces_v5_joint_convtasnet_micro_receipt_v1"
CHECKPOINT_FORMAT = "qces_v5_joint_convtasnet_micro_checkpoint_v1"
SAMPLE_RATE = 16_000
SAMPLES = 160_000


def parse_args() -> argparse.Namespace:
    data = Path("/var/tmp/qces_full188_tiered_realistic_v5")
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=data / "detector_scene_manifest_tiered_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=base / "v5_joint_convtasnet_micro_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8221)
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--num-sources", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--enc-num-feats", type=int, default=128)
    parser.add_argument("--mask-num-feats", type=int, default=64)
    parser.add_argument("--mask-hidden-feats", type=int, default=128)
    parser.add_argument("--mask-num-layers", type=int, default=6)
    parser.add_argument("--mask-num-stacks", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mono(path: Path) -> torch.Tensor:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    value = torch.from_numpy(np.asarray(audio)).float()
    if value.ndim > 1:
        value = value.mean(dim=-1)
    if int(rate) != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz: {path}")
    return F.pad(value[:SAMPLES], (0, max(0, SAMPLES - value.numel())))


@dataclass(frozen=True)
class SceneRef:
    row: dict[str, Any]

    @property
    def scene_id(self) -> str:
        return str(self.row["scene_id"])


def read_scenes(path: Path, maximum: int, num_sources: int) -> list[SceneRef]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if 1 < len(row["events"]) <= num_sources]
    # The micro gate must exercise actual simultaneous sources.  Select high
    # overlap scenes deterministically, not easy sparse examples.
    rows.sort(key=lambda row: (
        -float(row.get("actual_mean_event_overlap_fraction", 0.0)),
        hashlib.sha256(str(row["scene_id"]).encode()).hexdigest(),
    ))
    return [SceneRef(row) for row in rows[:maximum]]


class SceneDataset(Dataset[dict[str, Any]]):
    def __init__(self, scenes: Sequence[SceneRef], num_sources: int) -> None:
        self.scenes = list(scenes)
        self.num_sources = int(num_sources)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.scenes[index].row
        mixture = load_mono(Path(row["mixture_path"]))
        targets = torch.zeros(self.num_sources, SAMPLES)
        boxes = torch.zeros(self.num_sources, 2, dtype=torch.long)
        labels = [""] * self.num_sources
        overlap = torch.zeros(self.num_sources)
        for slot, event in enumerate(row["events"]):
            clip = load_mono(Path(event["component_path"]))
            clip_samples = int(event.get("source_active_samples", event.get("num_component_samples", clip.numel())))
            clip_samples = min(clip_samples + int(event.get("tail_zero_padding_samples", 0)), clip.numel())
            start = int(event.get("onset_sample", round(float(event["onset_seconds"]) * SAMPLE_RATE)))
            end = min(SAMPLES, start + clip_samples)
            targets[slot, start:end] = clip[: end - start]
            boxes[slot] = torch.tensor((start, end))
            labels[slot] = str(event["label"])
            overlap[slot] = float(event.get("overlap_fraction", 0.0))
        return {
            "mixture": mixture,
            "targets": targets,
            "boxes": boxes,
            "num_active": len(row["events"]),
            "scene_id": str(row["scene_id"]),
            "labels": labels,
            "overlap": overlap,
        }


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(dim=-1, keepdim=True) * target
    projection = projection / target.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    noise = estimate - projection
    return 10.0 * torch.log10(
        projection.square().sum(dim=-1).clamp_min(eps)
        / noise.square().sum(dim=-1).clamp_min(eps)
    )


def pairwise_si_sdr(prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return si_sdr(prediction[:, None, :], targets[None, :, :])


def match_active(prediction: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = pairwise_si_sdr(prediction, targets)
    output_rows, target_cols = linear_sum_assignment((-scores.detach().float().cpu().numpy()))
    return (
        torch.as_tensor(output_rows, device=prediction.device, dtype=torch.long),
        torch.as_tensor(target_cols, device=prediction.device, dtype=torch.long),
    )


def training_loss(
    prediction: torch.Tensor, mixture: torch.Tensor, targets: torch.Tensor, num_active: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    active_targets = targets[:num_active]
    output_indices, target_indices = match_active(prediction, active_targets)
    chosen_prediction = prediction[output_indices]
    chosen_targets = active_targets[target_indices]
    chosen_sisdr = si_sdr(chosen_prediction, chosen_targets).clamp(-30.0, 30.0)
    target_abs = chosen_targets.abs().mean(dim=-1).clamp_min(1e-5)
    normalized_l1 = ((chosen_prediction - chosen_targets).abs().mean(dim=-1) / target_abs).mean()
    rms_prediction = chosen_prediction.square().mean(dim=-1).sqrt().clamp_min(1e-6)
    rms_target = chosen_targets.square().mean(dim=-1).sqrt().clamp_min(1e-6)
    energy = torch.log(rms_prediction / rms_target).abs().mean()
    reconstruction = (prediction.sum(dim=0) - mixture).abs().mean() / mixture.abs().mean().clamp_min(1e-5)
    unmatched_mask = torch.ones(prediction.shape[0], device=prediction.device, dtype=torch.bool)
    unmatched_mask[output_indices] = False
    unmatched = prediction[unmatched_mask].square().sum() / mixture.square().sum().clamp_min(1e-6)
    loss = -chosen_sisdr.mean() + 2.0 * normalized_l1 + 0.5 * energy + 2.0 * reconstruction + 0.5 * unmatched
    return loss, {
        "pit_si_sdr_db": float(chosen_sisdr.detach().mean()),
        "normalized_l1": float(normalized_l1.detach()),
        "energy_log_error": float(energy.detach()),
        "mixture_reconstruction": float(reconstruction.detach()),
        "unmatched_energy_ratio": float(unmatched.detach()),
    }


def model_forward(model: torch.nn.Module, mixture: torch.Tensor, amp: bool) -> torch.Tensor:
    scale = mixture.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
    with torch.autocast("cuda", dtype=torch.float16, enabled=bool(amp and mixture.is_cuda)):
        prediction = model((mixture / scale)[:, None, :])
    return prediction.float() * scale[:, None, :]


def safe_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "positive_fraction": float((array > 0).mean()),
    }


@torch.inference_mode()
def evaluate(model: torch.nn.Module, dataset: SceneDataset, device: torch.device, amp: bool) -> dict[str, Any]:
    model.eval()
    improvements_mix: list[float] = []
    improvements_gate: list[float] = []
    overlap_improvements_gate: list[float] = []
    reconstruction_snr: list[float] = []
    items: list[dict[str, Any]] = []
    for sample in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        mixture = sample["mixture"].to(device)
        targets = sample["targets"][0].to(device)
        boxes = sample["boxes"][0].to(device)
        num_active = int(sample["num_active"].item())
        prediction = model_forward(model, mixture, amp)[0]
        output_indices, target_indices = match_active(prediction, targets[:num_active])
        scene_rows = []
        for output_index, target_index in zip(output_indices.tolist(), target_indices.tolist()):
            target = targets[target_index]
            estimate = prediction[output_index]
            start, end = boxes[target_index].tolist()
            gate_mixture = torch.zeros_like(mixture[0])
            gate_mixture[start:end] = mixture[0, start:end]
            predicted_score = float(si_sdr(estimate[None], target[None])[0])
            mixture_score = float(si_sdr(mixture, target[None])[0])
            gate_score = float(si_sdr(gate_mixture[None], target[None])[0])
            versus_mix = predicted_score - mixture_score
            versus_gate = predicted_score - gate_score
            overlap_fraction = float(sample["overlap"][0, target_index])
            improvements_mix.append(versus_mix)
            improvements_gate.append(versus_gate)
            if overlap_fraction >= 0.05:
                overlap_improvements_gate.append(versus_gate)
            scene_rows.append({
                "label": sample["labels"][target_index][0],
                "overlap_fraction": overlap_fraction,
                "predicted_si_sdr_db": predicted_score,
                "mixture_si_sdr_db": mixture_score,
                "oracle_span_mixture_si_sdr_db": gate_score,
                "si_sdri_vs_mixture_db": versus_mix,
                "si_sdri_vs_oracle_span_db": versus_gate,
            })
        error = prediction.sum(dim=0) - mixture[0]
        rec_snr = float(10.0 * torch.log10(mixture.square().sum() / error.square().sum().clamp_min(1e-8)))
        reconstruction_snr.append(rec_snr)
        items.append({"scene_id": sample["scene_id"][0], "reconstruction_snr_db": rec_snr, "events": scene_rows})
    result = {
        "scenes": len(dataset),
        "events": len(improvements_mix),
        "si_sdri_vs_raw_mixture_db_\u2191": safe_summary(improvements_mix),
        "si_sdri_vs_oracle_span_mixture_db_\u2191": safe_summary(improvements_gate),
        "overlap_events_si_sdri_vs_oracle_span_db_\u2191": safe_summary(overlap_improvements_gate) if overlap_improvements_gate else None,
        "mixture_reconstruction_snr_db_\u2191": safe_summary(reconstruction_snr),
        "items": items,
    }
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = read_scenes(args.manifest.resolve(), args.max_scenes, args.num_sources)
    if not scenes:
        raise ValueError("no eligible scenes")
    dataset = SceneDataset(scenes, args.num_sources)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torchaudio.models.ConvTasNet(
        num_sources=args.num_sources,
        enc_num_feats=args.enc_num_feats,
        msk_num_feats=args.mask_num_feats,
        msk_num_hidden_feats=args.mask_hidden_feats,
        msk_num_layers=args.mask_num_layers,
        msk_num_stacks=args.mask_num_stacks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    baseline = evaluate(model, dataset, device, args.amp)
    baseline_path = output_dir / "baseline_metrics.json"
    atomic_json(baseline, baseline_path)
    print(json.dumps({"baseline": {k: v for k, v in baseline.items() if k != "items"}}, sort_keys=True), flush=True)
    best_key = -math.inf
    best_epoch = 0
    best_metrics = baseline
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for sample in loader:
            mixture = sample["mixture"].to(device)
            targets = sample["targets"][0].to(device)
            num_active = int(sample["num_active"].item())
            prediction = model_forward(model, mixture, args.amp)[0]
            loss, details = training_loss(prediction, mixture[0], targets, num_active)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for key, value in details.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        scheduler.step()
        metrics = evaluate(model, dataset, device, args.amp)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {key: value / max(1, batches) for key, value in totals.items()},
            "micro": {key: value for key, value in metrics.items() if key != "items"},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = float(metrics["si_sdri_vs_raw_mixture_db_\u2191"]["median"])
        if key > best_key + 0.05:
            best_key = key
            best_epoch = epoch
            best_metrics = metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break
    checkpoint_path = output_dir / "joint_convtasnet_micro_v1_best.pt"
    atomic_torch({
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": best_state,
        "architecture": {
            "num_sources": args.num_sources,
            "enc_num_feats": args.enc_num_feats,
            "mask_num_feats": args.mask_num_feats,
            "mask_hidden_feats": args.mask_hidden_feats,
            "mask_num_layers": args.mask_num_layers,
            "mask_num_stacks": args.mask_num_stacks,
        },
        "best_epoch": best_epoch,
        "sample_rate": SAMPLE_RATE,
        "samples": SAMPLES,
    }, checkpoint_path)
    gates = {
        "median_si_sdri_vs_raw_mixture_ge_5dB": float(best_metrics["si_sdri_vs_raw_mixture_db_\u2191"]["median"]) >= 5.0,
        "positive_fraction_vs_raw_mixture_ge_0_90": float(best_metrics["si_sdri_vs_raw_mixture_db_\u2191"]["positive_fraction"]) >= 0.90,
        "overlap_median_not_worse_than_oracle_span": float(best_metrics["overlap_events_si_sdri_vs_oracle_span_db_\u2191"]["median"]) >= 0.0,
    }
    decision = "scale_joint_separator" if all(gates.values()) else "do_not_scale_joint_separator"
    receipt = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "micro-overfit capacity gate; not a paper result and not generalization evidence",
        "arguments": vars(args) | {"manifest": str(args.manifest.resolve()), "output_dir": str(output_dir)},
        "scene_ids": [scene.scene_id for scene in scenes],
        "scene_overlap": [float(scene.row.get("actual_mean_event_overlap_fraction", 0.0)) for scene in scenes],
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best": best_metrics,
        "history": history,
        "gates": gates,
        "decision": decision,
        "checkpoint": str(checkpoint_path),
    }
    receipt_path = output_dir / "training_receipt.json"
    atomic_json(receipt, receipt_path)
    receipt["artifacts"] = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "receipt": str(receipt_path),
        "baseline": str(baseline_path),
    }
    atomic_json(receipt, receipt_path)
    print(json.dumps({"receipt": str(receipt_path), "best_epoch": best_epoch, "gates": gates, "decision": decision}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
