#!/usr/bin/env python3
"""Train a class-agnostic temporal-span-conditioned separator on V4 stems."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.qces.temporal_tf_separator_v1 import (
    TemporalTFSeparatorV1, complex_multiply,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json, _atomic_torch, _device, _sha256_file,
)


FORMAT = "qces_temporal_tf_separator_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_temporal_tf_separator_checkpoint_v1"
SAMPLE_RATE = 16_000
SAMPLES = 160_000
N_FFT = 512
HOP = 256
WIN = 512


@dataclass(frozen=True)
class EventRef:
    scene_id: str
    mixture_path: str
    event: Mapping[str, Any]


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4"
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=data / "detector_scene_manifest_overlap_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=data / "detector_scene_manifest_overlap_dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=base / "temporal_tf_separator_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8221)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--max-dev-events", type=int, default=0)
    parser.add_argument("--overfit-train-as-dev", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_events(path: Path, maximum: int) -> list[EventRef]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: hashlib.sha256(str(row["scene_id"]).encode()).hexdigest())
    result = [
        EventRef(str(row["scene_id"]), str(row["mixture_path"]), event)
        for row in rows for event in row["events"]
    ]
    return result[:maximum] if maximum else result


def load_mono(path: Path) -> torch.Tensor:
    waveform, rate = sf.read(path, dtype="float32", always_2d=False)
    value = torch.as_tensor(waveform).float()
    if value.ndim > 1:
        value = value.mean(dim=-1)
    if int(rate) != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz: {path}")
    return value


class TemporalEventDataset(Dataset[dict[str, Any]]):
    def __init__(self, events: Sequence[EventRef], cache_size: int = 8) -> None:
        self.events = list(events)
        self.cache_size = int(cache_size)
        self._mixtures: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.events)

    def mixture(self, reference: EventRef) -> torch.Tensor:
        value = self._mixtures.get(reference.scene_id)
        if value is None:
            value = load_mono(Path(reference.mixture_path))
            value = F.pad(value[:SAMPLES], (0, max(0, SAMPLES - value.numel())))
            self._mixtures[reference.scene_id] = value
            if len(self._mixtures) > self.cache_size:
                self._mixtures.popitem(last=False)
        else:
            self._mixtures.move_to_end(reference.scene_id)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        reference = self.events[index]
        mixture = self.mixture(reference).clone()
        clip = load_mono(Path(str(reference.event["component_path"])))
        target = torch.zeros(SAMPLES)
        start = max(0, min(SAMPLES, int(round(float(reference.event["onset_seconds"]) * SAMPLE_RATE))))
        end = min(SAMPLES, start + clip.numel())
        target[start:end] = clip[: end - start]
        gate = torch.zeros(SAMPLES)
        gate[start:end] = 1.0
        return {
            "mixture": mixture, "target": target, "gate": gate,
            "event_id": str(reference.event["event_id"]),
            "label": str(reference.event["label"]),
        }


def stft(value: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        value.float(), n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=window, center=True, return_complex=True,
    )


def separator_forward(
    model: TemporalTFSeparatorV1, mixture: torch.Tensor, gate: torch.Tensor,
    window: torch.Tensor, *, amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = mixture.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
    normalized = mixture / scale
    mixture_spec = stft(normalized, window)
    magnitude = mixture_spec.abs().clamp_min(1e-7)
    log_magnitude = torch.log1p(magnitude)
    mean = log_magnitude.mean(dim=(-2, -1), keepdim=True)
    std = log_magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    normalized_log = (log_magnitude - mean) / std
    unit = mixture_spec / magnitude
    frame_gate = F.interpolate(gate[:, None], size=mixture_spec.shape[-1], mode="linear", align_corners=False)[:, 0]
    features = torch.stack((
        normalized_log, unit.real, unit.imag,
        frame_gate[:, None].expand(-1, mixture_spec.shape[-2], -1),
    ), dim=1)
    with torch.autocast(device_type=mixture.device.type, dtype=torch.float16, enabled=bool(amp and mixture.device.type == "cuda")):
        mask = model(features)
    prediction_spec = complex_multiply(mask, mixture_spec)
    prediction = torch.istft(
        prediction_spec, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=window, center=True, length=SAMPLES,
    )
    return prediction * scale, prediction_spec, mixture_spec, frame_gate


def loss_function(
    prediction: torch.Tensor, prediction_spec: torch.Tensor,
    target: torch.Tensor, target_spec: torch.Tensor,
    gate: torch.Tensor, frame_gate: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    on_frame = frame_gate[:, None]
    off_frame = 1.0 - on_frame
    complex_error = (prediction_spec - target_spec).abs()
    on_complex = (complex_error * on_frame).sum() / (on_frame.sum() * complex_error.shape[-2]).clamp_min(1)
    off_complex = (prediction_spec.abs() * off_frame).sum() / (off_frame.sum() * complex_error.shape[-2]).clamp_min(1)
    log_error = (torch.log1p(prediction_spec.abs()) - torch.log1p(target_spec.abs())).abs()
    on_log = (log_error * on_frame).sum() / (on_frame.sum() * log_error.shape[-2]).clamp_min(1)
    on_wave = ((prediction - target).abs() * gate).sum() / gate.sum().clamp_min(1)
    off_wave = (prediction.abs() * (1.0 - gate)).sum() / (1.0 - gate).sum().clamp_min(1)
    loss = on_complex + 0.10 * off_complex + 0.50 * on_log + 0.50 * on_wave + 0.10 * off_wave
    return loss, {
        "on_complex": float(on_complex.detach()), "off_complex": float(off_complex.detach()),
        "on_log": float(on_log.detach()), "on_wave": float(on_wave.detach()),
        "off_wave": float(off_wave.detach()),
    }


@torch.inference_mode()
def evaluate(
    model: TemporalTFSeparatorV1, dataset: TemporalEventDataset,
    device: torch.device, *, batch_size: int, amp: bool,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    window = torch.hann_window(WIN, device=device)
    improvements = []
    energy_ratios = []
    items = []
    model.eval()
    for batch in loader:
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        gate = batch["gate"].to(device)
        prediction, _, _, _ = separator_forward(model, mixture, gate, window, amp=amp)
        pred_si = scale_invariant_sdr(prediction, target)
        mix_si = scale_invariant_sdr(mixture, target)
        ratio = prediction.square().sum(dim=-1) / target.square().sum(dim=-1).clamp_min(1e-8)
        for index in range(len(batch["event_id"])):
            improvement = float(pred_si[index] - mix_si[index])
            improvements.append(improvement)
            energy_ratios.append(float(ratio[index]))
            items.append({"event_id": batch["event_id"][index], "label": batch["label"][index], "si_sdri_db": improvement, "energy_ratio": float(ratio[index])})
    values = torch.tensor(improvements)
    energies = torch.tensor(energy_ratios)
    return {
        "events": len(improvements),
        "si_sdri_mean_db_\u2191": float(values.mean()),
        "si_sdri_median_db_\u2191": float(values.median()),
        "si_sdri_positive_fraction_\u2191": float((values > 0).float().mean()),
        "energy_ratio_median_target_is_1": float(energies.median()),
        "items": items,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_refs = read_events(args.train_manifest.resolve(), args.max_train_events)
    dev_refs = train_refs if args.overfit_train_as_dev else read_events(args.dev_manifest.resolve(), args.max_dev_events)
    train_dataset = TemporalEventDataset(train_refs)
    dev_dataset = TemporalEventDataset(dev_refs)
    train_sources = {str(ref.event["source_id"]) for ref in train_refs}
    dev_sources = {str(ref.event["source_id"]) for ref in dev_refs}
    source_overlap = len(train_sources & dev_sources)
    if not args.overfit_train_as_dev and source_overlap:
        raise ValueError(f"train/dev source leakage: {source_overlap}")

    device = _device(args.device)
    model = TemporalTFSeparatorV1(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    window = torch.hann_window(WIN, device=device)
    baseline = evaluate(model, dev_dataset, device, batch_size=args.batch_size, amp=args.amp)
    best_metrics = baseline
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_key = float(best_metrics["si_sdri_median_db_\u2191"])
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            gate = batch["gate"].to(device)
            target_scale = mixture.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
            normalized_target = target / target_scale
            target_spec = stft(normalized_target, window)
            prediction, prediction_spec, _, frame_gate = separator_forward(model, mixture, gate, window, amp=args.amp)
            normalized_prediction = prediction / target_scale
            loss, _ = loss_function(
                normalized_prediction, prediction_spec, normalized_target,
                target_spec, gate, frame_gate,
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1
        scheduler.step()
        metrics = evaluate(model, dev_dataset, device, batch_size=args.batch_size, amp=args.amp)
        row = {"epoch": epoch, "train_loss": total_loss / max(batches, 1), "learning_rate": optimizer.param_groups[0]["lr"], "dev": {key: value for key, value in metrics.items() if key != "items"}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = float(metrics["si_sdri_median_db_\u2191"])
        if key > best_key + 0.05:
            best_key = key
            best_metrics = metrics
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    checkpoint_path = output_dir / "temporal_tf_separator_v1_best.pt"
    _atomic_torch({
        "format": CHECKPOINT_FORMAT, "model_state_dict": best_state,
        "base_channels": args.base_channels, "best_epoch": best_epoch,
        "stft": {"sample_rate": SAMPLE_RATE, "samples": SAMPLES, "n_fft": N_FFT, "hop": HOP, "win": WIN},
    }, checkpoint_path)
    if args.overfit_train_as_dev:
        gates = {
            "overfit_median_si_sdri_ge_5dB": float(best_metrics["si_sdri_median_db_\u2191"]) >= 5.0,
            "overfit_positive_fraction_ge_0_90": float(best_metrics["si_sdri_positive_fraction_\u2191"]) >= 0.90,
        }
        decision = "scale_temporal_separator" if all(gates.values()) else "separator_cannot_micro_overfit_fix_architecture"
    else:
        gates = {
            "dev_median_si_sdri_ge_2dB": float(best_metrics["si_sdri_median_db_\u2191"]) >= 2.0,
            "dev_positive_fraction_ge_0_70": float(best_metrics["si_sdri_positive_fraction_\u2191"]) >= 0.70,
        }
        decision = "evaluate_semantic_after_separation" if all(gates.values()) else "temporal_separator_insufficient"
    receipt = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "paper_eligible": False,
        "mode": "micro_overfit" if args.overfit_train_as_dev else "source_disjoint_development",
        "qa_question_or_answer_used_as_input": False,
        "method": "class_agnostic_mixture_plus_temporal_span_to_complex_mask_target_stem",
        "data": {"train_events": len(train_dataset), "dev_events": len(dev_dataset), "source_overlap": source_overlap},
        "baseline": {key: value for key, value in baseline.items() if key != "items"},
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "success_gates": gates, "decision": decision,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
        "inputs": {
            "train_manifest_sha256": _sha256_file(args.train_manifest.resolve()),
            "dev_manifest_sha256": _sha256_file(args.dev_manifest.resolve()),
        },
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "baseline": receipt["baseline"], "best_epoch": best_epoch, "best": {key: value for key, value in best_metrics.items() if key != "items"}, "gates": gates, "decision": decision}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
