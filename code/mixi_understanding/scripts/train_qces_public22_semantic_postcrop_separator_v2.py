#!/usr/bin/env python3
"""Train public-22 separation with semantic conditioning and post-crop timing.

This is the controlled architectural follow-up to robust-separator v1.  The
separator never receives the noisy detector span: it sees an all-one temporal
channel and the canonical CLAP text embedding.  The detector span is applied
only after semantic separation.  Separation supervision is exact-label only;
proxy rows remain in validation to expose, but not conceal, transfer failure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixi_understanding.qces.metrics import scale_dependent_sdr
from mixi_understanding.scripts.train_qces_public22_robust_separator_v1 import (
    RobustConditionDataset,
    _atomic_json,
    _read_jsonl,
    _seed,
    _sha256,
    _stratified_limit,
    _summary,
)
from mixi_understanding.scripts.train_qces_span_conditioned_separator_v1 import (
    CHECKPOINT_FORMAT,
    RATE,
    SpanMaskNetwork,
    _negative_si_sdr,
    _separator_forward,
    _stft,
)


FORMAT = "qces_public22_semantic_postcrop_separator_training_v2"


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_public22_separator_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=data / "event_components_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=data / "event_components_dev.jsonl")
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        default=PROJECT_ROOT / "outputs/qces_public22_robust_separator_v1_full/best.pt",
    )
    parser.add_argument(
        "--initial-label-cache", type=Path,
        default=PROJECT_ROOT / "outputs/qces_label_clap191_v1.pt",
    )
    parser.add_argument(
        "--label-embedding-cache", type=Path,
        default=PROJECT_ROOT / "outputs/qces_label_clap_public22_v1.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_public22_semantic_postcrop_separator_v2",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-seconds", type=float, default=2.56)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--exact-probability", type=float, default=0.15)
    parser.add_argument("--predicted-probability", type=float, default=0.55)
    parser.add_argument("--jittered-probability", type=float, default=0.30)
    parser.add_argument("--minimum-condition-iou", type=float, default=0.30)
    parser.add_argument("--train-pad-min-seconds", type=float, default=0.08)
    parser.add_argument("--train-pad-max-seconds", type=float, default=0.24)
    parser.add_argument("--validation-pad-seconds", type=float, default=0.15)
    parser.add_argument("--maximum-jitter-seconds", type=float, default=0.16)
    parser.add_argument("--soft-edge-seconds", type=float, default=0.04)
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--max-dev-events", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2251)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reset-output-layer", action=argparse.BooleanOptionalAction, default=True,
        help="Start the residual actuator at exact crop fallback after changing its temporal input.",
    )
    return parser.parse_args()


def _load_public22_model(
    checkpoint: Mapping[str, Any], initial_embeddings: torch.Tensor,
    public_embeddings: torch.Tensor,
) -> SpanMaskNetwork:
    config = checkpoint["config"]
    model = SpanMaskNetwork(
        base_channels=int(config["base_channels"]),
        semantic_channels=int(config["semantic_channels"]),
        label_embeddings=initial_embeddings.float(),
    )
    state = dict(checkpoint["model_state"])
    stored_embeddings = state.pop("label_embeddings")
    if not torch.equal(stored_embeddings.cpu(), public_embeddings.float().cpu()):
        raise ValueError("checkpoint label embeddings do not match the public-22 cache")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"label_embeddings"} or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.label_embeddings = public_embeddings.float().contiguous()
    return model


def _semantic_postcrop_forward(
    model: SpanMaskNetwork, mixture: torch.Tensor, output_span: torch.Tensor,
    label_id: torch.Tensor,
) -> torch.Tensor:
    # The separator condition is deliberately independent of detector timing.
    semantic_full, _, _, _ = _separator_forward(
        model, mixture, torch.ones_like(output_span), label_id
    )
    return semantic_full * output_span


def _per_sample_negative_si_sdr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (
        (prediction * target).sum(-1, keepdim=True)
        * target / (target.square().sum(-1, keepdim=True) + 1e-8)
    )
    noise = prediction - projection
    return -10.0 * torch.log10(
        (projection.square().sum(-1) + 1e-8) / (noise.square().sum(-1) + 1e-8)
    )


def _loss(
    model: SpanMaskNetwork, mixture: torch.Tensor, target: torch.Tensor,
    condition: torch.Tensor, gold: torch.Tensor, label_id: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = _semantic_postcrop_forward(model, mixture, condition, label_id)
    predicted_spec = _stft(prediction)
    target_spec = _stft(target)
    spectral_scale = target_spec.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    complex_per_item = F.smooth_l1_loss(
        torch.view_as_real(predicted_spec / spectral_scale),
        torch.view_as_real(target_spec / spectral_scale), beta=0.05, reduction="none",
    ).mean(dim=(1, 2, 3))
    logmag_per_item = F.l1_loss(
        torch.log1p(predicted_spec.abs()), torch.log1p(target_spec.abs()), reduction="none"
    ).mean(dim=(1, 2))
    wave_scale = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-3)
    waveform_per_item = F.smooth_l1_loss(
        prediction / wave_scale, target / wave_scale, beta=0.05, reduction="none"
    ).mean(dim=1)
    active_per_item = (
        ((prediction - target).abs() * gold).sum(-1)
        / (target.abs().sum(-1) + 1e-6)
    )
    leakage_per_item = (
        ((prediction * (1.0 - gold)).square().sum(-1) + 1e-8)
        / (target.square().sum(-1) + 1e-8)
    )
    energy_per_item = (
        (prediction.square().sum(-1) + 1e-8)
        / (target.square().sum(-1) + 1e-8)
    ).log().abs()
    negative_si_sdr_per_item = _per_sample_negative_si_sdr(prediction.float(), target.float())
    per_item = (
        complex_per_item + logmag_per_item + 0.50 * waveform_per_item
        + 0.25 * active_per_item + 0.20 * leakage_per_item
        + 0.10 * energy_per_item + 0.05 * negative_si_sdr_per_item
    )
    # Give the hardest item in each small batch modest extra weight.  This is
    # fixed a priori to address the measured catastrophic tail, not tuned on
    # individual classes.
    hardest = torch.topk(per_item, k=max(1, (per_item.numel() + 3) // 4)).values.mean()
    total = per_item.mean() + 0.20 * hardest
    return total, {
        "total": float(total.detach().cpu()),
        "complex": float(complex_per_item.mean().detach().cpu()),
        "waveform": float(waveform_per_item.mean().detach().cpu()),
        "active_recall_error": float(active_per_item.mean().detach().cpu()),
        "leakage_energy_ratio": float(leakage_per_item.mean().detach().cpu()),
        "negative_si_sdr": float(negative_si_sdr_per_item.mean().detach().cpu()),
        "tail_loss": float(hardest.detach().cpu()),
    }


def _subset(indices: Sequence[int], predicted: Sequence[float], baseline: Sequence[float]) -> dict[str, Any]:
    gains = [predicted[index] - baseline[index] for index in indices]
    return {
        "events": len(indices),
        "predicted_sd_sdr_db_↑": _summary([predicted[index] for index in indices]),
        "crop_baseline_sd_sdr_db_↑": _summary([baseline[index] for index in indices]),
        "sd_sdri_over_crop_db_↑": _summary(gains),
        "positive_rate_↑": float(np.mean(np.asarray(gains) > 0.0)) if gains else float("nan"),
        "harmful_below_minus_1db_rate_↓": (
            float(np.mean(np.asarray(gains) < -1.0)) if gains else float("nan")
        ),
    }


@torch.inference_mode()
def _validate(
    model: SpanMaskNetwork, loader: DataLoader, device: torch.device, labels: Sequence[str]
) -> dict[str, Any]:
    model.eval()
    predicted: list[float] = []
    baseline: list[float] = []
    ious: list[float] = []
    label_ids: list[int] = []
    proxy_flags: list[int] = []
    for mixture, target, condition, _gold, condition_iou, label_id, proxy, _mode in loader:
        mixture, target = mixture.to(device), target.to(device)
        condition, label_id_gpu = condition.to(device), label_id.to(device)
        prediction = _semantic_postcrop_forward(model, mixture, condition, label_id_gpu)
        predicted.extend(scale_dependent_sdr(prediction, target).cpu().tolist())
        baseline.extend(scale_dependent_sdr(mixture * condition, target).cpu().tolist())
        ious.extend(condition_iou.tolist())
        label_ids.extend(label_id.tolist())
        proxy_flags.extend(proxy.tolist())
    indices = list(range(len(predicted)))
    exact = [index for index in indices if proxy_flags[index] == 0]
    proxy = [index for index in indices if proxy_flags[index] == 1]
    return {
        "events": len(indices),
        "all": _subset(indices, predicted, baseline),
        "exact_semantics": _subset(exact, predicted, baseline),
        "proxy_semantics": _subset(proxy, predicted, baseline),
        "condition_iou": _summary(ious),
        "per_label": {
            labels[label_id]: _subset(
                [index for index in indices if label_ids[index] == label_id], predicted, baseline
            )
            for label_id in sorted(set(label_ids))
        },
    }


def _key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    exact = metrics["exact_semantics"]
    gain = exact["sd_sdri_over_crop_db_↑"]
    return (
        float(gain["median"]),
        float(exact["positive_rate_↑"]),
        float(gain["mean"]),
        -float(exact["harmful_below_minus_1db_rate_↓"]),
    )


def _save(
    path: Path, model: SpanMaskNetwork, epoch: int, metrics: Mapping[str, Any],
    config: Mapping[str, Any], labels: Sequence[str],
) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "training_format": FORMAT,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "metrics": dict(metrics),
        "config": dict(config),
        "labels": list(labels),
    }, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    _seed(args.seed)
    probability_sum = args.exact_probability + args.predicted_probability + args.jittered_probability
    if abs(probability_sum - 1.0) > 1e-6:
        raise ValueError(f"condition probabilities must sum to 1, got {probability_sum}")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    usable_train = [
        row for row in _read_jsonl(args.train_manifest.resolve())
        if bool(row.get("condition_available"))
        and float(row["condition_iou"]) >= args.minimum_condition_iou
    ]
    usable_dev = [
        row for row in _read_jsonl(args.dev_manifest.resolve())
        if bool(row.get("condition_available"))
        and float(row["condition_iou"]) >= args.minimum_condition_iou
    ]
    # Proxy labels are deliberately excluded from waveform supervision.
    train_rows = [row for row in usable_train if str(row.get("semantic_supervision")) == "exact"]
    train_rows = _stratified_limit(train_rows, args.max_train_events, args.seed)
    dev_rows = _stratified_limit(usable_dev, args.max_dev_events, args.seed + 1)

    public_cache = torch.load(args.label_embedding_cache.resolve(), map_location="cpu", weights_only=True)
    initial_cache = torch.load(args.initial_label_cache.resolve(), map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.initial_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("invalid initial separator checkpoint")
    labels = list(public_cache["labels"])
    model = _load_public22_model(
        checkpoint, initial_cache["embeddings"], public_cache["embeddings"]
    )
    if args.reset_output_layer:
        torch.nn.init.zeros_(model.output_layer.weight)
        if model.output_layer.bias is not None:
            torch.nn.init.zeros_(model.output_layer.bias)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    chunk_samples = int(round(args.chunk_seconds * RATE))
    dataset_kwargs = {
        "chunk_samples": chunk_samples,
        "exact_probability": args.exact_probability,
        "predicted_probability": args.predicted_probability,
        "pad_min": int(round(args.train_pad_min_seconds * RATE)),
        "pad_max": int(round(args.train_pad_max_seconds * RATE)),
        "validation_pad": int(round(args.validation_pad_seconds * RATE)),
        "maximum_jitter": int(round(args.maximum_jitter_seconds * RATE)),
        "soft_edge": int(round(args.soft_edge_seconds * RATE)),
    }
    train_dataset = RobustConditionDataset(train_rows, seed=args.seed, train=True, **dataset_kwargs)
    dev_dataset = RobustConditionDataset(dev_rows, seed=args.seed + 1, train=False, **dataset_kwargs)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    config = dict(checkpoint["config"])
    config.update({
        "conditioning": "public22_clap_text_semantic_only_then_soft_postcrop",
        "separator_temporal_input": "all_ones",
        "postcrop_temporal_input": "predicted_dilated_span",
        "waveform_supervision": "exact_semantics_only",
        "train_events": len(train_rows),
        "validation_events": len(dev_rows),
        "tail_objective": "mean_plus_0.20_worst_quartile_in_batch",
        "reset_output_layer": bool(args.reset_output_layer),
        "initial_checkpoint_sha256": _sha256(args.initial_checkpoint.resolve()),
        "public22_label_cache_sha256": _sha256(args.label_embedding_cache.resolve()),
        "selection": "exact-semantics median SD-SDRi over padded predicted crop",
        "acceptance_gate": {
            "median_gain_db_min": 1.0,
            "mean_gain_db_min": 0.0,
            "positive_rate_min": 0.60,
            "harmful_below_minus_1db_rate_max": 0.25,
        },
    })

    initial_metrics = _validate(model, dev_loader, device, labels)
    best_key = _key(initial_metrics)
    best_epoch = 0
    _save(output / "best.pt", model, 0, initial_metrics, config, labels)
    print(json.dumps({"epoch": 0, "validation": initial_metrics}, ensure_ascii=False), flush=True)
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        sums: dict[str, float] = {}
        steps = 0
        for step, batch in enumerate(train_loader, 1):
            mixture, target, condition, gold, _iou, label_id, _proxy, _mode = batch
            mixture, target = mixture.to(device), target.to(device)
            condition, gold, label_id = condition.to(device), gold.to(device), label_id.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss, terms = _loss(model, mixture, target, condition, gold, label_id)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + value
            if step % 100 == 0:
                print(json.dumps({
                    "epoch": epoch, "step": step, "steps": len(train_loader),
                    "mean_train_loss": sums["total"] / steps,
                }), flush=True)
        validation = _validate(model, dev_loader, device, labels)
        key = _key(validation)
        scheduler.step(key[0])
        record = {
            "epoch": epoch,
            "train": {name: value / steps for name, value in sums.items()},
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            _save(output / "best.pt", model, epoch, validation, config, labels)
        else:
            stale += 1
        if stale >= args.patience:
            break

    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    best_metrics = best["metrics"]
    exact = best_metrics["exact_semantics"]
    gain = exact["sd_sdri_over_crop_db_↑"]
    accepted = (
        float(gain["median"]) >= 1.0
        and float(gain["mean"]) >= 0.0
        and float(exact["positive_rate_↑"]) >= 0.60
        and float(exact["harmful_below_minus_1db_rate_↓"]) <= 0.25
    )
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "best_epoch": best_epoch,
        "accepted": accepted,
        "decision": "candidate_for_quality_headroom_audit" if accepted else "reject_semantic_postcrop_v2",
        "train_events": len(train_rows),
        "dev_events": len(dev_rows),
        "config": config,
        "best_validation": best_metrics,
        "history": history,
        "best_checkpoint": str((output / "best.pt").resolve()),
        "best_checkpoint_sha256": _sha256(output / "best.pt"),
        "train_manifest_sha256": _sha256(args.train_manifest.resolve()),
        "dev_manifest_sha256": _sha256(args.dev_manifest.resolve()),
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "complete": True,
        "best_epoch": best_epoch,
        "accepted": accepted,
        "decision": receipt["decision"],
        "exact_semantics": exact,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
