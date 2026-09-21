#!/usr/bin/env python3
"""Train a small instruction-conditioned removal model on overfit16.

This is intentionally a diagnostic baseline, not the final one-step model. It
answers one question: can the data path, conditioning path, model, loss, and
checkpoint path learn the 16 synthetic training pairs?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by setup only
    raise SystemExit(
        "PyTorch is required. Install code/mixi_understanding/requirements-train.txt "
        "inside .venv-data first."
    ) from exc


MODEL_DOWNSAMPLE_FACTOR = 4**4
EPSILON = 1e-8


@dataclass(frozen=True)
class TrainConfig:
    project_root: Path
    manifest: Path
    output_dir: Path
    epochs: int = 400
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    crop_seconds: float = 2.048
    focus_interference_probability: float = 0.8
    base_channels: int = 8
    embedding_dim: int = 64
    num_workers: int = 0
    seed: int = 2025
    device: str = "auto"
    grad_clip_norm: float = 5.0
    log_every_steps: int = 10
    eval_every_epochs: int = 10
    preview_count: int = 3
    max_steps: int = 0
    overwrite_output: bool = False


@dataclass
class RemovalExample:
    sample_id: str
    instruction: str
    event: str
    event_id: int
    sample_rate: int
    onset_sample: int
    offset_sample: int
    mixture: torch.Tensor
    target: torch.Tensor


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def read_mono_audio(path: Path) -> Tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        raise ValueError(f"Expected mono audio at {path}, got {waveform.shape[1]} channels")
    waveform = np.ascontiguousarray(waveform[:, 0], dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError(f"Audio contains NaN or Inf: {path}")
    return waveform, int(sample_rate)


class RemovalDataset(Dataset):
    """Preloads the tiny overfit split and returns event-focused waveform crops."""

    def __init__(
        self,
        manifest_path: Path,
        crop_samples: Optional[int],
        focus_interference_probability: float,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.dataset_root = self.manifest_path.parent.parent
        self.crop_samples = crop_samples
        self.focus_interference_probability = focus_interference_probability

        rows = read_jsonl(self.manifest_path)
        events = sorted({str(row["remove_event"]) for row in rows})
        self.event_to_id = {event: index for index, event in enumerate(events)}
        self.examples = [self._load_example(row) for row in rows]

        sample_rates = {example.sample_rate for example in self.examples}
        if len(sample_rates) != 1:
            raise ValueError(f"All examples must share one sample rate, got {sample_rates}")
        self.sample_rate = sample_rates.pop()
        if self.crop_samples is not None:
            if self.crop_samples < MODEL_DOWNSAMPLE_FACTOR:
                raise ValueError(
                    f"crop_samples must be at least {MODEL_DOWNSAMPLE_FACTOR}"
                )
            shortest = min(example.mixture.numel() for example in self.examples)
            if self.crop_samples > shortest:
                raise ValueError(
                    f"crop_samples={self.crop_samples} exceeds shortest clip={shortest}"
                )

    def _load_example(self, row: Mapping[str, Any]) -> RemovalExample:
        required = (
            "id",
            "instruction",
            "remove_event",
            "sample_rate",
            "num_samples",
            "interference_onset_seconds",
            "interference_offset_seconds",
            "mixture_path",
            "target_path",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Manifest row is missing keys {missing}: {row.get('id')}")

        mixture_path = (self.dataset_root / str(row["mixture_path"])).resolve()
        target_path = (self.dataset_root / str(row["target_path"])).resolve()
        mixture, mixture_rate = read_mono_audio(mixture_path)
        target, target_rate = read_mono_audio(target_path)
        expected_rate = int(row["sample_rate"])
        expected_samples = int(row["num_samples"])
        if mixture_rate != expected_rate or target_rate != expected_rate:
            raise ValueError(
                f"Sample-rate mismatch for {row['id']}: "
                f"manifest={expected_rate}, mixture={mixture_rate}, target={target_rate}"
            )
        if mixture.shape != target.shape or mixture.size != expected_samples:
            raise ValueError(
                f"Shape mismatch for {row['id']}: mixture={mixture.shape}, "
                f"target={target.shape}, expected={expected_samples}"
            )

        onset = int(round(float(row["interference_onset_seconds"]) * expected_rate))
        offset = int(round(float(row["interference_offset_seconds"]) * expected_rate))
        onset = max(0, min(onset, expected_samples - 1))
        offset = max(onset + 1, min(offset, expected_samples))
        event = str(row["remove_event"])
        return RemovalExample(
            sample_id=str(row["id"]),
            instruction=str(row["instruction"]),
            event=event,
            event_id=self.event_to_id[event],
            sample_rate=expected_rate,
            onset_sample=onset,
            offset_sample=offset,
            mixture=torch.from_numpy(mixture),
            target=torch.from_numpy(target),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def _sample_crop_start(self, example: RemovalExample) -> int:
        assert self.crop_samples is not None
        total_samples = example.mixture.numel()
        max_start = total_samples - self.crop_samples
        if max_start <= 0:
            return 0

        focus_event = bool(
            torch.rand(()) < self.focus_interference_probability
        )
        if not focus_event:
            return int(torch.randint(0, max_start + 1, ()).item())

        event_sample = int(
            torch.randint(example.onset_sample, example.offset_sample, ()).item()
        )
        minimum = max(0, event_sample - self.crop_samples + 1)
        maximum = min(event_sample, max_start)
        return int(torch.randint(minimum, maximum + 1, ()).item())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        if self.crop_samples is None:
            start = 0
            stop = example.mixture.numel()
        else:
            start = self._sample_crop_start(example)
            stop = start + self.crop_samples
        return {
            "id": example.sample_id,
            "instruction": example.instruction,
            "event": example.event,
            "event_id": torch.tensor(example.event_id, dtype=torch.long),
            "mixture": example.mixture[start:stop],
            "target": example.target[start:stop],
        }


def group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class FeatureBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=5, padding=2),
            nn.GroupNorm(group_count(output_channels), output_channels),
            nn.SiLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=5, padding=2),
            nn.GroupNorm(group_count(output_channels), output_channels),
            nn.SiLU(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.layers(waveform)


class DownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=8,
                stride=4,
                padding=2,
            ),
            nn.GroupNorm(group_count(output_channels), output_channels),
            nn.SiLU(),
        )
        self.features = FeatureBlock(output_channels, output_channels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.features(self.down(waveform))


class ConditionalWaveUNet(nn.Module):
    """Small Wave-U-Net with event conditioning at its bottleneck."""

    def __init__(
        self,
        num_events: int,
        base_channels: int = 8,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 6,
        )
        self.down1 = DownBlock(1, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)
        self.down4 = DownBlock(c3, c4)
        self.bottleneck = FeatureBlock(c4, c4)

        self.event_embedding = nn.Embedding(num_events, embedding_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, c4 * 2),
        )

        self.up3 = nn.ConvTranspose1d(c4, c3, kernel_size=8, stride=4, padding=2)
        self.decode3 = FeatureBlock(c3 + c3, c3)
        self.up2 = nn.ConvTranspose1d(c3, c2, kernel_size=8, stride=4, padding=2)
        self.decode2 = FeatureBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose1d(c2, c1, kernel_size=8, stride=4, padding=2)
        self.decode1 = FeatureBlock(c1 + c1, c1)
        self.residual_head = nn.ConvTranspose1d(
            c1, 1, kernel_size=8, stride=4, padding=2
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self, mixture: torch.Tensor, event_ids: torch.Tensor
    ) -> torch.Tensor:
        if mixture.ndim != 2:
            raise ValueError(f"Expected mixture [batch, samples], got {mixture.shape}")
        original_samples = mixture.shape[-1]
        padding = (-original_samples) % MODEL_DOWNSAMPLE_FACTOR
        network_input = F.pad(mixture, (0, padding)).unsqueeze(1)

        encoded1 = self.down1(network_input)
        encoded2 = self.down2(encoded1)
        encoded3 = self.down3(encoded2)
        encoded4 = self.down4(encoded3)
        latent = self.bottleneck(encoded4)

        condition = self.condition_projection(self.event_embedding(event_ids))
        scale, shift = condition.chunk(2, dim=1)
        latent = latent * (1.0 + torch.tanh(scale).unsqueeze(-1))
        latent = latent + shift.unsqueeze(-1)

        decoded3 = self.decode3(torch.cat((self.up3(latent), encoded3), dim=1))
        decoded2 = self.decode2(torch.cat((self.up2(decoded3), encoded2), dim=1))
        decoded1 = self.decode1(torch.cat((self.up1(decoded2), encoded1), dim=1))
        residual = torch.tanh(self.residual_head(decoded1).squeeze(1))
        estimate = network_input.squeeze(1) + residual
        return estimate[:, :original_samples]


def waveform_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = F.l1_loss(estimate, target)
    estimate_channel = estimate.unsqueeze(1)
    target_channel = target.unsqueeze(1)
    for factor, weight in ((4, 0.5), (16, 0.25), (64, 0.125)):
        pooled_estimate = F.avg_pool1d(
            estimate_channel, kernel_size=factor, stride=factor
        )
        pooled_target = F.avg_pool1d(
            target_channel, kernel_size=factor, stride=factor
        )
        loss = loss + weight * F.l1_loss(pooled_estimate, pooled_target)
    return loss


def si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection_scale = (estimate * target).sum(dim=-1, keepdim=True) / (
        target.square().sum(dim=-1, keepdim=True) + EPSILON
    )
    projection = projection_scale * target
    noise = estimate - projection
    ratio = projection.square().sum(dim=-1) / (
        noise.square().sum(dim=-1) + EPSILON
    )
    return 10.0 * torch.log10(ratio + EPSILON)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_config(config: TrainConfig) -> Dict[str, Any]:
    result = asdict(config)
    for key in ("project_root", "manifest", "output_dir"):
        result[key] = str(result[key])
    return result


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def evaluate(
    model: nn.Module,
    dataset: RemovalDataset,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    mixture_l1_values: List[float] = []
    estimate_l1_values: List[float] = []
    mixture_sisdr_values: List[float] = []
    estimate_sisdr_values: List[float] = []
    with torch.inference_mode():
        for example in dataset.examples:
            mixture = example.mixture.unsqueeze(0).to(device)
            target = example.target.unsqueeze(0).to(device)
            event_id = torch.tensor([example.event_id], device=device)
            estimate = model(mixture, event_id)
            mixture_l1_values.append(F.l1_loss(mixture, target).item())
            estimate_l1_values.append(F.l1_loss(estimate, target).item())
            mixture_sisdr_values.append(si_sdr(mixture, target).item())
            estimate_sisdr_values.append(si_sdr(estimate, target).item())

    mixture_l1 = float(np.mean(mixture_l1_values))
    estimate_l1 = float(np.mean(estimate_l1_values))
    mixture_sisdr = float(np.mean(mixture_sisdr_values))
    estimate_sisdr = float(np.mean(estimate_sisdr_values))
    return {
        "mixture_l1": mixture_l1,
        "estimate_l1": estimate_l1,
        "l1_reduction_percent": 100.0 * (mixture_l1 - estimate_l1) / mixture_l1,
        "mixture_si_sdr_db": mixture_sisdr,
        "estimate_si_sdr_db": estimate_sisdr,
        "si_sdr_improvement_db": estimate_sisdr - mixture_sisdr,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    dataset: RemovalDataset,
    epoch: int,
    step: int,
    metrics: Mapping[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": serializable_config(config),
            "event_to_id": dataset.event_to_id,
            "epoch": epoch,
            "step": step,
            "metrics": dict(metrics),
        },
        path,
    )


def save_previews(
    model: nn.Module,
    dataset: RemovalDataset,
    device: torch.device,
    output_dir: Path,
    count: int,
) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for example in dataset.examples[:count]:
            mixture = example.mixture.unsqueeze(0).to(device)
            event_id = torch.tensor([example.event_id], device=device)
            estimate = model(mixture, event_id)[0].cpu().numpy()
            sample_dir = preview_dir / example.sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            sf.write(
                sample_dir / "mixture.wav",
                example.mixture.numpy(),
                example.sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                sample_dir / "target.wav",
                example.target.numpy(),
                example.sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                sample_dir / "estimate.wav",
                np.clip(estimate, -1.0, 1.0),
                example.sample_rate,
                subtype="PCM_16",
            )
            (sample_dir / "instruction.txt").write_text(
                example.instruction + "\n", encoding="utf-8"
            )


def prepare_output_directory(config: TrainConfig) -> None:
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not config.overwrite_output:
            raise FileExistsError(
                f"Output directory is not empty: {config.output_dir}. "
                "Choose another --output-dir or pass --overwrite-output."
            )
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)


def train_overfit(config: TrainConfig) -> Dict[str, Any]:
    """Train the diagnostic removal model and return the final run summary."""

    set_seed(config.seed)
    prepare_output_directory(config)
    device = choose_device(config.device)

    manifest_rows = read_jsonl(config.manifest)
    sample_rate = int(manifest_rows[0]["sample_rate"])
    requested_crop = int(round(config.crop_seconds * sample_rate))
    crop_samples = max(
        MODEL_DOWNSAMPLE_FACTOR,
        (requested_crop // MODEL_DOWNSAMPLE_FACTOR) * MODEL_DOWNSAMPLE_FACTOR,
    )
    dataset = RemovalDataset(
        config.manifest,
        crop_samples=crop_samples,
        focus_interference_probability=config.focus_interference_probability,
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        drop_last=False,
    )

    model = ConditionalWaveUNet(
        num_events=len(dataset.event_to_id),
        base_channels=config.base_channels,
        embedding_dim=config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    config_payload = serializable_config(config)
    config_payload.update(
        {
            "actual_crop_samples": crop_samples,
            "actual_crop_seconds": crop_samples / dataset.sample_rate,
            "device_resolved": str(device),
            "event_to_id": dataset.event_to_id,
            "parameter_count": parameter_count,
            "torch_version": torch.__version__,
        }
    )
    (config.output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = config.output_dir / "metrics.jsonl"

    print(
        f"Loaded {len(dataset)} examples, {len(dataset.event_to_id)} events, "
        f"{parameter_count:,} parameters on {device}."
    )
    initial_metrics = evaluate(model, dataset, device)
    append_jsonl(
        metrics_path,
        {"kind": "evaluation", "epoch": 0, "step": 0, **initial_metrics},
    )
    print(
        "Before training: "
        f"L1={initial_metrics['estimate_l1']:.6f}, "
        f"SI-SDR={initial_metrics['estimate_si_sdr_db']:.3f} dB"
    )
    best_l1 = initial_metrics["estimate_l1"]
    best_metrics = initial_metrics
    save_checkpoint(
        config.output_dir / "best.pt",
        model,
        optimizer,
        config,
        dataset,
        epoch=0,
        step=0,
        metrics=initial_metrics,
    )

    global_step = 0
    last_epoch = 0
    latest_train_loss = math.nan
    stop_requested = False
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for batch in loader:
            mixture = batch["mixture"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            event_ids = batch["event_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            estimate = model(mixture, event_ids)
            loss = waveform_loss(estimate, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step + 1}")
            loss.backward()
            gradient_norm = clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            global_step += 1
            latest_train_loss = float(loss.item())
            epoch_losses.append(latest_train_loss)
            if global_step == 1 or global_step % config.log_every_steps == 0:
                print(
                    f"epoch={epoch:04d} step={global_step:05d} "
                    f"loss={latest_train_loss:.6f} grad_norm={float(gradient_norm):.4f}"
                )
            if config.max_steps and global_step >= config.max_steps:
                stop_requested = True
                break

        last_epoch = epoch
        mean_epoch_loss = float(np.mean(epoch_losses))
        append_jsonl(
            metrics_path,
            {
                "kind": "training",
                "epoch": epoch,
                "step": global_step,
                "mean_loss": mean_epoch_loss,
            },
        )
        should_evaluate = (
            epoch == 1
            or epoch % config.eval_every_epochs == 0
            or epoch == config.epochs
            or stop_requested
        )
        if should_evaluate:
            current_metrics = evaluate(model, dataset, device)
            append_jsonl(
                metrics_path,
                {
                    "kind": "evaluation",
                    "epoch": epoch,
                    "step": global_step,
                    **current_metrics,
                },
            )
            print(
                f"Evaluation: L1={current_metrics['estimate_l1']:.6f} "
                f"({current_metrics['l1_reduction_percent']:+.2f}% vs mixture), "
                f"SI-SDRi={current_metrics['si_sdr_improvement_db']:+.3f} dB"
            )
            if current_metrics["estimate_l1"] < best_l1:
                best_l1 = current_metrics["estimate_l1"]
                best_metrics = current_metrics
                save_checkpoint(
                    config.output_dir / "best.pt",
                    model,
                    optimizer,
                    config,
                    dataset,
                    epoch=epoch,
                    step=global_step,
                    metrics=current_metrics,
                )
        if stop_requested:
            break

    final_metrics = evaluate(model, dataset, device)
    save_checkpoint(
        config.output_dir / "last.pt",
        model,
        optimizer,
        config,
        dataset,
        epoch=last_epoch,
        step=global_step,
        metrics=final_metrics,
    )

    best_checkpoint_path = config.output_dir / "best.pt"
    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    save_previews(
        model,
        dataset,
        device,
        config.output_dir,
        min(config.preview_count, len(dataset)),
    )

    summary: Dict[str, Any] = {
        "status": "completed",
        "examples": len(dataset),
        "events": len(dataset.event_to_id),
        "parameters": parameter_count,
        "device": str(device),
        "epochs_completed": last_epoch,
        "steps_completed": global_step,
        "latest_train_loss": latest_train_loss,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(config.output_dir / "last.pt"),
        "metrics_log": str(metrics_path),
        "preview_checkpoint": str(best_checkpoint_path),
        "preview_directory": str(config.output_dir / "previews"),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Finished: "
        f"initial L1={initial_metrics['estimate_l1']:.6f}, "
        f"final L1={final_metrics['estimate_l1']:.6f}, "
        f"reduction={final_metrics['l1_reduction_percent']:+.2f}%"
    )
    return summary


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Overfit a small conditional removal model on 16 examples."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/removal_synthetic_v1/manifests/overfit16.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/removal_overfit16"),
    )
    parser.add_argument("--epochs", type=positive_int, default=400)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--crop-seconds", type=positive_float, default=2.048)
    parser.add_argument(
        "--focus-interference-probability", type=probability, default=0.8
    )
    parser.add_argument("--base-channels", type=positive_int, default=8)
    parser.add_argument("--embedding-dim", type=positive_int, default=64)
    parser.add_argument("--num-workers", type=non_negative_int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-clip-norm", type=positive_float, default=5.0)
    parser.add_argument("--log-every-steps", type=positive_int, default=10)
    parser.add_argument("--eval-every-epochs", type=positive_int, default=10)
    parser.add_argument("--preview-count", type=non_negative_int, default=3)
    parser.add_argument(
        "--max-steps",
        type=non_negative_int,
        default=0,
        help="Stop after this many optimizer steps; 0 means no limit.",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    project_root = args.project_root.resolve()
    return TrainConfig(
        project_root=project_root,
        manifest=resolve_path(args.manifest, project_root),
        output_dir=resolve_path(args.output_dir, project_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        crop_seconds=args.crop_seconds,
        focus_interference_probability=args.focus_interference_probability,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        grad_clip_norm=args.grad_clip_norm,
        log_every_steps=args.log_every_steps,
        eval_every_epochs=args.eval_every_epochs,
        preview_count=args.preview_count,
        max_steps=args.max_steps,
        overwrite_output=args.overwrite_output,
    )


def main() -> None:
    train_overfit(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
