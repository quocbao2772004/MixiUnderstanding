#!/usr/bin/env python3
"""Cache target-free quality features for public-22 evidence candidates.

The script intentionally keeps candidate generation, deployable feature
extraction, and offline quality labels separate:

* candidate waveforms use only mixture, predicted temporal condition, and the
  requested public label;
* deployable features use waveform agreement plus a frozen public BEATs-Strong
  detector re-score;
* the clean target is consulted only to create SD-SDR gain labels for training
  and evaluation of a later selector.

No checkpoint used by the current public demo is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader

from mixi_understanding.qces.metrics import scale_dependent_sdr
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    encode_prompts,
)
from mixi_understanding.scripts.evaluate_qces_public22_audiosep_postcrop_v1 import (
    AUDIOSEP_RATE,
    PROMPTS,
)
from mixi_understanding.scripts.infer_qces_combined_honest_v1 import (
    RAW_TO_LABEL,
    SPEECH_RAW,
)
from mixi_understanding.scripts.train_qces_public22_robust_separator_v1 import (
    RobustConditionDataset,
    _read_jsonl,
    _stratified_limit,
)
from mixi_understanding.scripts.train_qces_span_conditioned_separator_v1 import (
    CHECKPOINT_FORMAT,
    RATE,
    SpanMaskNetwork,
    _separator_forward,
)


FORMAT = "qces_public22_selector_features_v2"
CANDIDATES = ("crop", "custom_span_postcrop", "audiosep_projected")
DETECTOR_SECONDS = 10.0
DETECTOR_FRAME_SECONDS = 0.04
DETECTOR_MEDIAN_FRAMES = 9


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_public22_separator_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=data / "event_components_dev.jsonl")
    parser.add_argument(
        "--custom-checkpoint", type=Path,
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
    parser.add_argument("--audiosep-root", type=Path, default=CODE_ROOT / "baseline/audiosep")
    parser.add_argument(
        "--audiosep-config", type=Path,
        default=CODE_ROOT / "baseline/audiosep/config/audiosep_base.yaml",
    )
    parser.add_argument(
        "--audiosep-checkpoint", type=Path,
        default=CODE_ROOT / "baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_public22_selector_features_v2_dev",
    )
    parser.add_argument("--chunk-seconds", type=float, default=2.56)
    parser.add_argument("--minimum-condition-iou", type=float, default=0.30)
    parser.add_argument("--validation-pad-seconds", type=float, default=0.15)
    parser.add_argument("--soft-edge-seconds", type=float, default=0.04)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2261)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _load_custom(args: argparse.Namespace, device: torch.device) -> tuple[SpanMaskNetwork, list[str]]:
    checkpoint = torch.load(args.custom_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("invalid public-22 custom separator checkpoint")
    initial_cache = torch.load(args.initial_label_cache.resolve(), map_location="cpu", weights_only=True)
    public_cache = torch.load(args.label_embedding_cache.resolve(), map_location="cpu", weights_only=True)
    labels = list(public_cache["labels"])
    config = checkpoint["config"]
    model = SpanMaskNetwork(
        base_channels=int(config["base_channels"]),
        semantic_channels=int(config["semantic_channels"]),
        label_embeddings=initial_cache["embeddings"].float(),
    )
    model_state = dict(checkpoint["model_state"])
    stored_embeddings = model_state.pop("label_embeddings")
    if not torch.equal(stored_embeddings.cpu(), public_cache["embeddings"].float().cpu()):
        raise ValueError("checkpoint embeddings do not match public-22 cache")
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if set(missing) != {"label_embeddings"} or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.label_embeddings = public_cache["embeddings"].float().contiguous()
    return model.to(device).eval(), labels


def _load_detector(device: torch.device):
    import models.prediction_wrapper as prediction_wrapper
    from data_util.audioset_classes import as_strong_train_classes
    from models.beats.BEATs_wrapper import BEATsWrapper
    from models.prediction_wrapper import PredictionsWrapper

    prediction_wrapper.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    detector = PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
    detector.eval().to(device)
    label_to_id = {label: index for index, label in enumerate(as_strong_train_classes)}
    required = set(RAW_TO_LABEL) | set(SPEECH_RAW)
    missing = sorted(required - set(label_to_id))
    if missing:
        raise RuntimeError(f"public detector missing labels: {missing}")
    public_to_raw: dict[str, list[str]] = {}
    for raw, public in RAW_TO_LABEL.items():
        public_to_raw.setdefault(public, []).append(raw)
    public_to_raw["Speech"] = list(SPEECH_RAW)
    return detector, label_to_id, public_to_raw


def _safe_log(value: torch.Tensor, floor: float = 1e-8) -> torch.Tensor:
    return torch.log10(value.clamp_min(floor))


def _waveform_features(candidates: Mapping[str, torch.Tensor]) -> list[dict[str, float]]:
    crop = candidates["crop"].float()
    custom = candidates["custom_span_postcrop"].float()
    audiosep = candidates["audiosep_projected"].float()
    batch = crop.shape[0]
    eps = 1e-8

    def stats(value: torch.Tensor) -> dict[str, torch.Tensor]:
        energy = value.square().sum(-1)
        rms = value.square().mean(-1).sqrt()
        peak = value.abs().amax(-1)
        crest = peak / rms.clamp_min(1e-5)
        zcr = ((value[:, 1:] * value[:, :-1]) < 0).float().mean(-1)
        spectrum = torch.fft.rfft(value, dim=-1).abs().clamp_min(1e-8)
        flatness = torch.exp(torch.log(spectrum).mean(-1)) / spectrum.mean(-1).clamp_min(1e-8)
        frequencies = torch.linspace(0.0, 1.0, spectrum.shape[-1], device=value.device)
        centroid = (spectrum * frequencies).sum(-1) / spectrum.sum(-1).clamp_min(1e-8)
        return {
            "energy": energy, "rms": rms, "peak": peak, "crest": crest,
            "zcr": zcr, "flatness": flatness, "centroid": centroid,
        }

    all_stats = {name: stats(value) for name, value in candidates.items()}

    def relation(left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        left_energy = left.square().sum(-1)
        right_energy = right.square().sum(-1)
        correlation = (left * right).sum(-1) / (
            left_energy.sqrt() * right_energy.sqrt()
        ).clamp_min(eps)
        difference = (left - right).square().sum(-1) / right_energy.clamp_min(eps)
        energy_ratio = left_energy / right_energy.clamp_min(eps)
        return {
            "correlation": correlation,
            "difference_energy_ratio": difference,
            "energy_ratio": energy_ratio,
        }

    versus_crop = {
        name: relation(value, crop)
        for name, value in candidates.items() if name != "crop"
    }
    agreement = relation(custom, audiosep)
    rows: list[dict[str, float]] = []
    for index in range(batch):
        row: dict[str, float] = {}
        for name, values in all_stats.items():
            for key in ("rms", "peak", "crest", "zcr", "flatness", "centroid"):
                value = values[key][index]
                row[f"wave_{name}_{key}"] = float(value.cpu())
            row[f"wave_{name}_log_energy"] = float(_safe_log(values["energy"][index]).cpu())
        for name, values in versus_crop.items():
            row[f"wave_{name}_crop_correlation"] = float(values["correlation"][index].cpu())
            row[f"wave_{name}_crop_log_difference_energy_ratio"] = float(
                _safe_log(values["difference_energy_ratio"][index]).cpu()
            )
            row[f"wave_{name}_crop_log_energy_ratio"] = float(
                _safe_log(values["energy_ratio"][index]).cpu()
            )
        row["wave_custom_audiosep_correlation"] = float(agreement["correlation"][index].cpu())
        row["wave_custom_audiosep_log_difference_energy_ratio"] = float(
            _safe_log(agreement["difference_energy_ratio"][index]).cpu()
        )
        row["wave_custom_audiosep_log_energy_ratio"] = float(
            _safe_log(agreement["energy_ratio"][index]).cpu()
        )
        rows.append(row)
    return rows


@torch.inference_mode()
def _detector_features(
    detector: torch.nn.Module,
    label_to_id: Mapping[str, int],
    public_to_raw: Mapping[str, Sequence[str]],
    candidates: Mapping[str, torch.Tensor],
    labels: Sequence[str],
    condition: torch.Tensor,
    device: torch.device,
) -> list[dict[str, float]]:
    batch = condition.shape[0]
    candidate_audio = torch.stack([candidates[name] for name in CANDIDATES], dim=1)
    flat = candidate_audio.reshape(batch * len(CANDIDATES), -1)
    target_samples = int(round(DETECTOR_SECONDS * RATE))
    flat = flat[:, :target_samples]
    if flat.shape[-1] < target_samples:
        flat = F.pad(flat, (0, target_samples - flat.shape[-1]))
    mel = detector.mel_forward(flat.to(device))
    strong_logits, weak_logits = detector(mel)
    strong = strong_logits.sigmoid().transpose(1, 2).float().cpu().numpy()
    weak = weak_logits.sigmoid().float().cpu().numpy()
    public_labels = sorted(public_to_raw)
    raw_ids = {
        public: [label_to_id[raw] for raw in raw_labels]
        for public, raw_labels in public_to_raw.items()
    }
    valid_frames = min(
        strong.shape[1], int(math.ceil(candidate_audio.shape[-1] / RATE / DETECTOR_FRAME_SECONDS))
    )
    selected_raw = sorted({index for indices in raw_ids.values() for index in indices})
    reduced = strong[:, :valid_frames, selected_raw]
    reduced = median_filter(reduced, size=(1, DETECTOR_MEDIAN_FRAMES, 1), mode="nearest")
    reduced_lookup = {raw_id: index for index, raw_id in enumerate(selected_raw)}
    public_frame = np.zeros((flat.shape[0], valid_frames, len(public_labels)), dtype=np.float32)
    public_weak = np.zeros((flat.shape[0], len(public_labels)), dtype=np.float32)
    for public_index, public in enumerate(public_labels):
        ids = raw_ids[public]
        columns = [reduced_lookup[index] for index in ids]
        public_frame[:, :, public_index] = reduced[:, :, columns].max(axis=2)
        public_weak[:, public_index] = weak[:, ids].max(axis=1)

    condition_frames = F.interpolate(
        condition[:, None].float(), size=valid_frames, mode="linear", align_corners=False
    )[:, 0].cpu().numpy()
    condition_active = condition_frames >= 0.25
    rows: list[dict[str, float]] = []
    public_to_index = {label: index for index, label in enumerate(public_labels)}
    for sample_index, label in enumerate(labels):
        if label not in public_to_index:
            raise ValueError(f"label has no public detector mapping: {label}")
        target_index = public_to_index[label]
        feature_row: dict[str, float] = {}
        for candidate_index, candidate_name in enumerate(CANDIDATES):
            flat_index = sample_index * len(CANDIDATES) + candidate_index
            frames = public_frame[flat_index]
            target_values = frames[:, target_index]
            other_mask = np.arange(len(public_labels)) != target_index
            other_values = frames[:, other_mask]
            class_peak = frames.max(axis=0)
            class_mean = frames.mean(axis=0)
            target_peak = float(target_values.max())
            target_mean = float(target_values.mean())
            top_count = min(5, len(target_values))
            target_top5 = float(np.partition(target_values, -top_count)[-top_count:].mean())
            distractor_peak = float(class_peak[other_mask].max())
            distractor_mean = float(class_mean[other_mask].max())
            target_rank = float(1 + np.sum(class_peak[other_mask] > target_peak))
            active = condition_active[sample_index]
            if np.any(active):
                active_mean = float(target_values[active].mean())
                active_peak = float(target_values[active].max())
            else:
                active_mean = target_mean
                active_peak = target_peak
            if np.any(~active):
                outside_mean = float(target_values[~active].mean())
            else:
                outside_mean = 0.0
            prefix = f"beats_{candidate_name}"
            feature_row.update({
                f"{prefix}_target_peak": target_peak,
                f"{prefix}_target_mean": target_mean,
                f"{prefix}_target_top5_mean": target_top5,
                f"{prefix}_target_std": float(target_values.std()),
                f"{prefix}_target_weak": float(public_weak[flat_index, target_index]),
                f"{prefix}_target_active_peak": active_peak,
                f"{prefix}_target_active_mean": active_mean,
                f"{prefix}_target_outside_mean": outside_mean,
                f"{prefix}_target_temporal_contrast": active_mean - outside_mean,
                f"{prefix}_max_distractor_peak": distractor_peak,
                f"{prefix}_max_distractor_mean": distractor_mean,
                f"{prefix}_target_peak_margin": target_peak - distractor_peak,
                f"{prefix}_target_mean_margin": target_mean - distractor_mean,
                f"{prefix}_target_peak_rank_fraction": target_rank / len(public_labels),
            })
        for candidate_name in CANDIDATES[1:]:
            for statistic in (
                "target_peak", "target_mean", "target_top5_mean", "target_weak",
                "target_peak_margin", "target_mean_margin",
            ):
                feature_row[f"beats_{candidate_name}_minus_crop_{statistic}"] = (
                    feature_row[f"beats_{candidate_name}_{statistic}"]
                    - feature_row[f"beats_crop_{statistic}"]
                )
        rows.append(feature_row)
    return rows


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    all_rows = _read_jsonl(args.manifest.resolve())
    rows = [
        row for row in all_rows
        if bool(row.get("condition_available"))
        and float(row["condition_iou"]) >= args.minimum_condition_iou
    ]
    rows = _stratified_limit(rows, args.max_events, args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    custom, labels = _load_custom(args, device)
    missing_prompts = sorted(set(labels) - set(PROMPTS))
    if missing_prompts:
        raise ValueError(f"missing AudioSep prompts: {missing_prompts}")
    prompt_embeddings = encode_prompts(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(),
        [PROMPTS[label] for label in labels], batch_size=len(labels),
    )
    audiosep = _load_separator(
        SimpleNamespace(
            audiosep_root=args.audiosep_root,
            audiosep_config=args.audiosep_config,
            audiosep_checkpoint=args.audiosep_checkpoint,
        ),
        device,
    )
    detector, label_to_id, public_to_raw = _load_detector(device)
    dataset = RobustConditionDataset(
        rows,
        chunk_samples=int(round(args.chunk_seconds * RATE)),
        seed=args.seed + 1,
        train=False,
        exact_probability=0.0,
        predicted_probability=1.0,
        pad_min=0,
        pad_max=0,
        validation_pad=int(round(args.validation_pad_seconds * RATE)),
        maximum_jitter=0,
        soft_edge=int(round(args.soft_edge_seconds * RATE)),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    items: list[dict[str, Any]] = []
    cursor = 0
    for batch in loader:
        mixture, target, condition, _gold, condition_iou, label_id, proxy, _mode = batch
        mixture = mixture.to(device)
        target = target.to(device)
        condition_gpu = condition.to(device)
        label_gpu = label_id.to(device)
        custom_raw, _, _, _ = _separator_forward(custom, mixture, condition_gpu, label_gpu)
        crop = mixture * condition_gpu
        custom_postcrop = custom_raw * condition_gpu

        mixture_32k = AF.resample(mixture, RATE, AUDIOSEP_RATE)
        batch_labels = [labels[int(value)] for value in label_id.tolist()]
        embedding = torch.stack(
            [prompt_embeddings[PROMPTS[label]] for label in batch_labels]
        ).to(device)
        raw_32k = audiosep({"mixture": mixture_32k[:, None], "condition": embedding})["waveform"][:, 0]
        raw = AF.resample(raw_32k, AUDIOSEP_RATE, RATE)
        if raw.shape[-1] != mixture.shape[-1]:
            raw = F.interpolate(raw[:, None], size=mixture.shape[-1], mode="linear", align_corners=False)[:, 0]
        audiosep_postcrop = raw * condition_gpu
        projection_scale = (
            (audiosep_postcrop * crop).sum(-1, keepdim=True)
            / audiosep_postcrop.square().sum(-1, keepdim=True).clamp_min(1e-8)
        ).clamp(0.25, 4.0)
        audiosep_projected = audiosep_postcrop * projection_scale
        candidates = {
            "crop": crop,
            "custom_span_postcrop": custom_postcrop,
            "audiosep_projected": audiosep_projected,
        }
        waveform_rows = _waveform_features(candidates)
        detector_rows = _detector_features(
            detector, label_to_id, public_to_raw, candidates, batch_labels,
            condition_gpu, device,
        )
        scores = {
            name: scale_dependent_sdr(value, target).float().cpu().tolist()
            for name, value in candidates.items()
        }
        for local_index in range(mixture.shape[0]):
            source = rows[cursor]
            score_row = {name: float(value[local_index]) for name, value in scores.items()}
            features: dict[str, float] = {
                "condition_score": float(source.get("condition_score") or 0.0),
                "condition_score_squared": float(source.get("condition_score") or 0.0) ** 2,
                "condition_active_fraction": float((condition[local_index] >= 0.25).float().mean()),
                "audiosep_projection_scale": float(projection_scale[local_index, 0].cpu()),
                **waveform_rows[local_index],
                **detector_rows[local_index],
            }
            crop_score = score_row["crop"]
            items.append({
                "event_id": str(source["event_id"]),
                "scene_id": str(source["scene_id"]),
                "source_id": str(source["source_id"]),
                "label": batch_labels[local_index],
                "label_id": int(label_id[local_index]),
                "semantic_supervision": str(source.get("semantic_supervision", "unknown")),
                "is_proxy": bool(int(proxy[local_index])),
                "features": features,
                "offline_quality_labels": {
                    "sd_sdr_db": score_row,
                    "gain_over_crop_db": {
                        name: float(score_row[name] - crop_score)
                        for name in CANDIDATES[1:]
                    },
                },
            })
            cursor += 1
        if cursor % 25 < mixture.shape[0] or cursor == len(rows):
            print(json.dumps({"done": cursor, "total": len(rows)}), flush=True)

    exact = [row for row in items if not row["is_proxy"]]
    summary: dict[str, Any] = {}
    for candidate in CANDIDATES[1:]:
        gains = [row["offline_quality_labels"]["gain_over_crop_db"][candidate] for row in exact]
        summary[candidate] = {
            "gain_over_crop_db_↑": _summary(gains),
            "positive_rate_↑": float(np.mean(np.asarray(gains) > 0.0)),
            "harmful_below_minus_1db_rate_↓": float(np.mean(np.asarray(gains) < -1.0)),
        }
    items_path = output / "items.jsonl"
    _write_jsonl(items_path, items)
    feature_names = sorted(items[0]["features"]) if items else []
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "events": len(items),
        "exact_events": len(exact),
        "proxy_events": len(items) - len(exact),
        "candidates": CANDIDATES,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "uses_target_at_inference": False,
        "condition_iou_is_feature": False,
        "offline_target_use": "SD-SDR candidate gain labels only",
        "exact_candidate_summary": summary,
        "inputs": {
            "manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256(args.manifest.resolve())},
            "custom_checkpoint": {"path": str(args.custom_checkpoint.resolve()), "sha256": _sha256(args.custom_checkpoint.resolve())},
            "audiosep_checkpoint": {"path": str(args.audiosep_checkpoint.resolve()), "sha256": _sha256(args.audiosep_checkpoint.resolve())},
            "beats_checkpoint": {
                "path": str(PRETRAINED_SED_ROOT / "resources/BEATs_strong_1.pt"),
                "sha256": _sha256(PRETRAINED_SED_ROOT / "resources/BEATs_strong_1.pt"),
            },
        },
        "items": str(items_path.resolve()),
        "items_sha256": _sha256(items_path),
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "complete": True,
        "events": len(items),
        "feature_count": len(feature_names),
        "exact_candidate_summary": summary,
        "receipt": str((output / "receipt.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
