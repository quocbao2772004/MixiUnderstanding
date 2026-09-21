#!/usr/bin/env python3
"""Train the fixed-grid 200-class QCES BEATs-Strong detector.

This is a new sidecar trainer.  It intentionally does not reuse the temporal
target construction from the historical detector trainer.  Its contract is:

* 250 left-aligned frames at exactly 40 ms, independent of clip duration;
* every official, ontology-selected strong event in a scene is a positive;
* right-padded frames never contribute to loss or metrics;
* Stage 1 trains only the native-initialized 200-class strong head;
* optional Stage 2 opens exactly the last two BEATs transformer blocks; and
* paper-scale runs require a hashed quota/integrity gate receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import torch
import torchaudio
import torchaudio.functional as audio_functional
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import models.prediction_wrapper as pretrained_prediction_wrapper
from mixi_understanding.qces.clean_detector_splits import audit_strict_identity_overlap
from mixi_understanding.qces.fixed_grid_detector import (
    FIXED_AUDIO_SECONDS,
    FRAME_HOP_SECONDS,
    NUM_FRAMES,
    balancing_tensors,
    boundary_pos_weights,
    boundary_probabilities_from_activity_logits,
    build_fixed_grid_boundary_targets,
    build_fixed_grid_targets,
    class_coverage,
    evaluate_fixed_grid_predictions,
    masked_balanced_bce,
    masked_probability_bce,
    valid_frames_for_duration,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    interpolate_sequence,
    load_model,
)


FORMAT = "qces_supported_beats_detector_training_v2"
CHECKPOINT_FORMAT = "qces_supported_beats_detector_checkpoint_v2"
GATE_FORMAT = "qces_supported_detector_training_gate_v2"
NATIVE_INITIALIZATION_FORMAT = "qces_native_supported_detector_200_v1"

DEFAULT_PROTOCOL_ROOT = PROJECT_ROOT / "outputs/qces_supported_clean_protocol_v2_current_audio"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1/ontology_200_supported.txt"
DEFAULT_INITIALIZATION = (
    PROJECT_ROOT / "outputs/qces_native_supported_detector_200_v1/native_supported_detector.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/qces_supported_beats_detector_200_v2"

pretrained_prediction_wrapper.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")


@dataclass(frozen=True)
class DetectorScene:
    scene_id: str
    split: str
    mixture_path: str
    duration_seconds: float
    sample_rate: int
    events: tuple[dict[str, Any], ...]
    raw: Mapping[str, Any]


class DetectorAudioDataset(Dataset[tuple[torch.Tensor, DetectorScene]]):
    def __init__(
        self,
        rows: Sequence[DetectorScene],
        *,
        audio_root: Path,
        sample_rate: int = 16_000,
    ) -> None:
        self.rows = list(rows)
        self.audio_root = audio_root
        self.sample_rate = int(sample_rate)
        self.fixed_samples = int(round(FIXED_AUDIO_SECONDS * self.sample_rate))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectorScene]:
        row = self.rows[index]
        path = Path(row.mixture_path)
        if not path.is_absolute():
            path = self.audio_root / path
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float().mean(dim=0) if waveform.ndim == 2 else waveform.float().reshape(-1)
        if int(sample_rate) != self.sample_rate:
            waveform = audio_functional.resample(waveform, int(sample_rate), self.sample_rate)
        required_samples = int(math.ceil(row.duration_seconds * self.sample_rate - 1e-6))
        if waveform.numel() + 2 < required_samples:
            raise ValueError(
                f"audio for {row.scene_id} is shorter than declared duration: "
                f"samples={waveform.numel()} required={required_samples}"
            )
        if waveform.numel() < self.fixed_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.fixed_samples - waveform.numel()))
        else:
            waveform = waveform[: self.fixed_samples]
        return waveform, row


def collate_audio(
    batch: Sequence[tuple[torch.Tensor, DetectorScene]],
) -> tuple[torch.Tensor, list[DetectorScene]]:
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_PROTOCOL_ROOT / "detector_manifest_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=DEFAULT_PROTOCOL_ROOT / "detector_manifest_dev.jsonl")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_PROTOCOL_ROOT / "detector_manifest_test.jsonl")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--initialization", type=Path, default=DEFAULT_INITIALIZATION)
    parser.add_argument(
        "--gate-receipt",
        type=Path,
        default=DEFAULT_PROTOCOL_ROOT / "detector_training_gate_v2.json",
        help=f"Required {GATE_FORMAT} receipt unless --allow-debug-data is set.",
    )
    parser.add_argument("--audio-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-head-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-backbone-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=80.0)
    parser.add_argument("--max-boundary-pos-weight", type=float, default=200.0)
    parser.add_argument("--onset-loss-weight", type=float, default=0.5)
    parser.add_argument("--offset-loss-weight", type=float, default=0.5)
    parser.add_argument("--boundary-dilation-frames", type=int, default=0)
    parser.add_argument("--boundary-diagnostic-threshold", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--min-event-duration", type=float, default=0.08)
    parser.add_argument("--merge-gap", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--allow-debug-data", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 200 or len(set(labels)) != 200:
        raise ValueError(f"ontology must contain exactly 200 unique labels: {path}")
    return labels


def load_manifest(
    path: Path,
    labels: Sequence[str],
    *,
    expected_split: str,
    max_scenes: int = 0,
) -> tuple[list[DetectorScene], list[dict[str, Any]]]:
    """Load all scored official events; never collapse a scene to one label."""

    label_to_id = {label: index for index, label in enumerate(labels)}
    rows: list[DetectorScene] = []
    raw_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in _read_jsonl(path):
        scene_id = str(raw.get("scene_id") or "").strip()
        split = str(raw.get("protocol_split") or raw.get("split") or "").strip()
        if not scene_id:
            raise ValueError(f"manifest row without scene_id: {path}")
        if scene_id in seen:
            raise ValueError(f"duplicate scene_id in {path}: {scene_id}")
        if split and split != expected_split:
            raise ValueError(f"scene {scene_id} has split={split!r}, expected {expected_split!r}")
        mixture_path = str(raw.get("mixture_path") or "").strip()
        duration = float(raw.get("duration_seconds", FIXED_AUDIO_SECONDS))
        sample_rate = int(raw.get("sample_rate", 16_000))
        if not mixture_path:
            raise ValueError(f"scene {scene_id} has no mixture_path")
        if not np.isfinite(duration) or duration <= 0.0 or duration > FIXED_AUDIO_SECONDS:
            raise ValueError(
                f"scene {scene_id} duration must lie in (0,{FIXED_AUDIO_SECONDS}]"
            )
        if sample_rate <= 0:
            raise ValueError(f"scene {scene_id} has invalid sample_rate={sample_rate}")
        events: list[dict[str, Any]] = []
        for event in raw.get("events") or ():
            label = str(event.get("label") or "").strip()
            if label not in label_to_id:
                raise ValueError(
                    f"scored event {label!r} is outside the 200-class ontology in scene {scene_id}; "
                    "non-target official events must be stored in context_events"
                )
            onset = float(event.get("onset_seconds", 0.0))
            offset = float(event.get("offset_seconds", 0.0))
            if (
                not np.isfinite(onset)
                or not np.isfinite(offset)
                or onset < 0.0
                or offset <= onset
                or offset > duration + FRAME_HOP_SECONDS + 1e-8
            ):
                raise ValueError(f"invalid event interval in scene {scene_id}: {event}")
            # Official timestamps are often rounded independently from decoded
            # sample duration.  Permit at most one 40-ms grid frame of mismatch
            # and make the repair explicit in the normalized event; larger
            # overruns remain fatal rather than silently creating padding labels.
            clipped_offset = min(offset, duration)
            if clipped_offset <= onset:
                raise ValueError(f"event disappears after duration clipping in {scene_id}: {event}")
            events.append(
                {
                    **event,
                    "label": label,
                    "label_id": label_to_id[label],
                    "onset_seconds": onset,
                    "offset_seconds": clipped_offset,
                    "raw_offset_seconds": offset,
                    "offset_clipped_to_duration": bool(offset > duration),
                }
            )
        # A selected class may not be silently downgraded to context: that
        # would turn a true positive into a false negative during training.
        leaked_context = sorted(
            {
                str(event.get("label") or "").strip()
                for event in raw.get("context_events") or ()
                if str(event.get("label") or "").strip() in label_to_id
            }
        )
        if leaked_context:
            raise ValueError(f"selected labels incorrectly stored as context in {scene_id}: {leaked_context}")
        row = DetectorScene(
            scene_id=scene_id,
            split=expected_split,
            mixture_path=mixture_path,
            duration_seconds=duration,
            sample_rate=sample_rate,
            events=tuple(events),
            raw=raw,
        )
        rows.append(row)
        raw_rows.append(raw)
        seen.add(scene_id)
        if max_scenes > 0 and len(rows) >= max_scenes:
            break
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows, raw_rows


def split_calibration_selection_rows(
    rows: Sequence[DetectorScene],
    *,
    seed: int,
    calibration_fraction: float,
) -> tuple[list[DetectorScene], list[DetectorScene]]:
    """Deterministically split development scenes before threshold tuning."""

    if len(rows) < 2:
        raise ValueError("at least two development scenes are required")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0,1)")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:beats-v2-calibration:{row.scene_id}".encode("utf-8")
        ).hexdigest(),
    )
    count = min(
        len(ranked) - 1,
        max(1, int(round(len(ranked) * calibration_fraction))),
    )
    calibration, selection = ranked[:count], ranked[count:]
    if {row.scene_id for row in calibration} & {row.scene_id for row in selection}:
        raise RuntimeError("calibration and selection scene split overlaps")
    return calibration, selection


def validate_gate_receipt(
    path: Path,
    *,
    ontology: Path,
    manifests: Mapping[str, Path],
    allow_debug_data: bool,
) -> dict[str, Any]:
    """Validate the exact data gate; a missing/failed receipt is fatal."""

    if allow_debug_data:
        return {
            "format": "debug_override",
            "passes": False,
            "paper_eligible": False,
            "reason": "--allow-debug-data bypassed the mandatory quota/integrity receipt",
        }
    if not path.is_file():
        raise RuntimeError(
            f"missing detector quota/integrity gate receipt: {path}; "
            "run the supported-detector gate audit or use --allow-debug-data only for smoke tests"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != GATE_FORMAT:
        raise RuntimeError(f"unexpected gate format {payload.get('format')!r}; expected {GATE_FORMAT}")
    if payload.get("passes") is not True or payload.get("paper_eligible") is not True:
        raise RuntimeError(f"detector data gate failed: {path}")
    ontology_entry = payload.get("ontology") or {}
    if int(ontology_entry.get("labels", 0)) != 200 or ontology_entry.get("sha256") != _sha256(ontology):
        raise RuntimeError("gate ontology hash/count does not match the requested ontology")
    receipt_manifests = payload.get("manifests") or {}
    for split, manifest in manifests.items():
        entry = receipt_manifests.get(split) or {}
        if entry.get("sha256") != _sha256(manifest):
            raise RuntimeError(f"gate manifest hash mismatch for {split}: {manifest}")
    quota = payload.get("quota") or {}
    if (
        quota.get("passes") is not True
        or int(quota.get("selected_classes", 0)) != 200
        or int(quota.get("classes_meeting_train_eval_targets", 0)) != 200
        or int(quota.get("train_target_videos_per_label", 0)) < 100
        or int(quota.get("eval_target_videos_per_label", 0)) < 20
    ):
        raise RuntimeError("gate does not prove 200/200 classes meet the 100-train/20-eval quota")
    integrity = payload.get("integrity") or {}
    if integrity.get("passes") is not True or int(integrity.get("cross_split_hard_overlap", -1)) != 0:
        raise RuntimeError("gate does not prove source-disjoint manifests")
    return payload


def validate_native_initialization(
    payload: Mapping[str, Any],
    *,
    labels: Sequence[str],
    ontology: Path,
) -> dict[str, Any]:
    """Verify that checkpoint channel ``i`` really maps to ontology label ``i``."""

    if payload.get("format") != NATIVE_INITIALIZATION_FORMAT:
        raise RuntimeError(
            f"unexpected native initialization format: {payload.get('format')!r}"
        )
    if list(payload.get("labels") or []) != list(labels):
        raise RuntimeError("initialization checkpoint labels do not match ontology order")
    initialization = payload.get("initialization") or {}
    if initialization.get("ontology_sha256") != _sha256(ontology):
        raise RuntimeError("native initialization ontology hash does not match")
    native_count = int(initialization.get("native_head_labels", 0))
    rows = initialization.get("selected_native_rows")
    if (
        native_count < len(labels)
        or not isinstance(rows, list)
        or len(rows) != len(labels)
        or len(set(int(value) for value in rows)) != len(labels)
        or any(int(value) < 0 or int(value) >= native_count for value in rows)
    ):
        raise RuntimeError("native initialization row mapping is missing or invalid")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("native initialization has no model_state_dict")
    for key in (
        "strong_head.weight",
        "strong_head.bias",
        "weak_head.weight",
        "weak_head.bias",
    ):
        tensor = state.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[0] != len(labels):
            raise RuntimeError(f"native initialization head tensor is invalid: {key}")
    source_hash = str(initialization.get("checkpoint_sha256") or "")
    if len(source_hash) != 64:
        raise RuntimeError("native initialization does not identify its source checkpoint")
    return {
        "format": NATIVE_INITIALIZATION_FORMAT,
        "native_head_labels": native_count,
        "selected_rows": len(rows),
        "selected_rows_unique": True,
        "ontology_sha256_matches": True,
        "source_checkpoint_sha256": source_hash,
        "passes": True,
    }


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_logits(model: torch.nn.Module, waveforms: torch.Tensor) -> torch.Tensor:
    backbone_trainable = any(parameter.requires_grad for parameter in model.model.parameters())
    context = contextlib.nullcontext() if backbone_trainable else torch.no_grad()
    with context:
        mel = model.mel_forward(waveforms)
        features = model.model(mel)
        features = interpolate_sequence(features, model.seq_len)
        features = model.seq_model(features)
    return model.strong_head(features)


def configure_stage(model: torch.nn.Module, stage: str) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Configure head-only Stage 1 or last-two-block Stage 2."""

    model.requires_grad_(False)
    model.strong_head.requires_grad_(True)
    head = list(model.strong_head.parameters())
    backbone: list[torch.nn.Parameter] = []
    if stage == "stage2_last_two_blocks":
        for name, parameter in model.model.named_parameters():
            if (
                "beats.encoder.layers.10." in name
                or "beats.encoder.layers.11." in name
            ):
                parameter.requires_grad = True
                backbone.append(parameter)
        trainable_names = {
            name for name, parameter in model.model.named_parameters() if parameter.requires_grad
        }
        if not any("beats.encoder.layers.10." in name for name in trainable_names) or not any(
            "beats.encoder.layers.11." in name for name in trainable_names
        ):
            raise RuntimeError("could not locate both BEATs encoder layers 10/11 for Stage 2")
        unexpected = sorted(
            name
            for name in trainable_names
            if "beats.encoder.layers.10." not in name
            and "beats.encoder.layers.11." not in name
        )
        if unexpected:
            raise RuntimeError(f"Stage 2 unexpectedly unfroze backbone parameters: {unexpected[:5]}")
    elif stage != "stage1_head_only":
        raise ValueError(f"unknown training stage: {stage}")
    return head, backbone


def set_train_mode_without_frozen_backbone_dropout(model: torch.nn.Module) -> None:
    """Enable stochastic layers only inside modules that are actually tuned.

    Calling ``model.train()`` during Stage 2 would also enable dropout and
    layer-drop in frozen BEATs blocks 0--9.  Those activations would change on
    every pass even though their parameters are fixed.  Start from a fully
    deterministic eval backbone, then explicitly enable the strong head and
    the two trainable transformer blocks.
    """

    model.eval()
    model.strong_head.train()
    trainable_module_names = {
        "beats.encoder.layers.10",
        "beats.encoder.layers.11",
    }
    enabled: set[str] = set()
    for name, module in model.model.named_modules():
        if name in trainable_module_names and any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.train()
            enabled.add(name)
    backbone_trainable = any(parameter.requires_grad for parameter in model.model.parameters())
    if backbone_trainable and not {
        "beats.encoder.layers.10",
        "beats.encoder.layers.11",
    }.issubset(enabled):
        raise RuntimeError(
            "Stage 2 has trainable backbone parameters but the last two BEATs blocks "
            "could not be placed in train mode"
        )


def _make_train_loader(
    rows: Sequence[DetectorScene],
    *,
    audio_root: Path,
    sampler_weights: torch.Tensor,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        sampler_weights,
        num_samples=len(rows),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        DetectorAudioDataset(rows, audio_root=audio_root),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_audio,
    )


def _make_eval_loader(
    rows: Sequence[DetectorScene],
    *,
    audio_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        DetectorAudioDataset(rows, audio_root=audio_root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_audio,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    pos_weight: torch.Tensor,
    onset_pos_weight: torch.Tensor,
    offset_pos_weight: torch.Tensor,
    class_weight: torch.Tensor,
    device: torch.device,
    amp: bool,
    gradient_clip: float,
    onset_loss_weight: float,
    offset_loss_weight: float,
    boundary_dilation_frames: int,
    trainable_parameters: Sequence[torch.nn.Parameter],
) -> float:
    set_train_mode_without_frozen_backbone_dropout(model)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    losses: list[float] = []
    for waveforms, rows in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = _forward_logits(model, waveforms)
            if logits.shape[1:] != (NUM_FRAMES, len(pos_weight)):
                raise RuntimeError(f"unexpected detector logits shape: {tuple(logits.shape)}")
            targets, valid_mask = build_fixed_grid_targets(
                rows, num_labels=len(pos_weight), device=device
            )
            frame_loss = masked_balanced_bce(
                logits,
                targets,
                valid_mask,
                pos_weight=pos_weight,
                class_weight=class_weight,
            )
            onset_targets, offset_targets, boundary_valid = build_fixed_grid_boundary_targets(
                rows,
                num_labels=len(pos_weight),
                dilation_frames=boundary_dilation_frames,
                device=device,
            )
            if not torch.equal(valid_mask, boundary_valid):
                raise RuntimeError("activity and boundary target padding masks disagree")
            onset_probability, offset_probability = boundary_probabilities_from_activity_logits(
                logits
            )
            onset_loss = masked_probability_bce(
                onset_probability,
                onset_targets,
                valid_mask,
                pos_weight=onset_pos_weight,
                class_weight=class_weight,
            )
            offset_loss = masked_probability_bce(
                offset_probability,
                offset_targets,
                valid_mask,
                pos_weight=offset_pos_weight,
                class_weight=class_weight,
            )
            loss = (
                frame_loss
                + float(onset_loss_weight) * onset_loss
                + float(offset_loss_weight) * offset_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_parameters, gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, *, device: torch.device, amp: bool
) -> list[dict[str, Any]]:
    model.eval()
    output: list[dict[str, Any]] = []
    for waveforms, rows in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = _forward_logits(model, waveforms)
        logits = logits.detach().cpu().to(torch.float16)
        for index, row in enumerate(rows):
            output.append(
                {
                    "scene_id": row.scene_id,
                    "duration_seconds": row.duration_seconds,
                    "valid_frames": valid_frames_for_duration(row.duration_seconds),
                    "gold_events": list(row.events),
                    "logits": logits[index],
                }
            )
    return output


def evaluate_thresholds(
    predictions: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str],
    thresholds: Sequence[float],
    event_iou_threshold: float,
    min_duration_seconds: float,
    merge_gap_seconds: float,
    boundary_diagnostic_threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    summaries = [
        evaluate_fixed_grid_predictions(
            predictions,
            labels=labels,
            threshold=threshold,
            event_iou_threshold=event_iou_threshold,
            min_duration_seconds=min_duration_seconds,
            merge_gap_seconds=merge_gap_seconds,
            boundary_diagnostic_threshold=boundary_diagnostic_threshold,
        )
        for threshold in thresholds
    ]
    selected = max(
        summaries,
        key=lambda row: (
            float(row["event"]["macro_f1_observed_classes"]),
            float(row["event"]["f1"]),
            float(row["frame"]["macro_f1_observed_classes"]),
            float(row["frame"]["f1"]),
            -abs(float(row["threshold"]) - 0.5),
        ),
    )
    curve = [
        {
            "threshold": float(row["threshold"]),
            "frame_f1": float(row["frame"]["f1"]),
            "event_f1": float(row["event"]["f1"]),
            "event_macro_f1": float(row["event"]["macro_f1_observed_classes"]),
            "oracle_window_top1": float(row["oracle_window_exact_label"]["top1_micro"]),
            "oracle_window_top5": float(row["oracle_window_exact_label"]["top5_micro"]),
        }
        for row in summaries
    ]
    return selected, curve


def _selection_score(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    oracle = metrics["oracle_window_exact_label"]
    return (
        float(metrics["event"]["macro_f1_observed_classes"]),
        float(metrics["event"]["f1"]),
        float(metrics["frame"]["macro_f1_observed_classes"]),
        float(metrics["frame"]["f1"]),
        float(oracle["top1_micro"]),
    )


def _compact_epoch(
    *,
    stage: str,
    epoch: int,
    loss: float | None,
    calibration_metrics: Mapping[str, Any],
    selection_metrics: Mapping[str, Any],
    curve: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    oracle = selection_metrics["oracle_window_exact_label"]
    return {
        "stage": stage,
        "epoch": epoch,
        "train_loss": loss,
        "selected_threshold": calibration_metrics["threshold"],
        "threshold_calibrated_on": "calibration_scenes_only",
        "checkpoint_selected_on": "selection_scenes_only",
        "oracle_window_exact_label_top1_↑": oracle["top1_micro"],
        "oracle_window_exact_label_top5_↑": oracle["top5_micro"],
        "frame_f1_↑": selection_metrics["frame"]["f1"],
        "event_f1_↑": selection_metrics["event"]["f1"],
        "event_macro_f1_↑": selection_metrics["event"]["macro_f1_observed_classes"],
        "onset_frame_f1_↑": selection_metrics["onset_frame"]["f1"],
        "offset_frame_f1_↑": selection_metrics["offset_frame"]["f1"],
        "calibration_threshold_curve": list(curve),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.stage1_epochs < 0 or args.stage2_epochs < 0:
        raise ValueError("stage epoch counts must be non-negative")
    if not args.thresholds or any(not 0.0 <= value <= 1.0 for value in args.thresholds):
        raise ValueError("thresholds must be a non-empty list in [0,1]")
    if len(set(float(value) for value in args.thresholds)) != len(args.thresholds):
        raise ValueError("thresholds must be unique")
    if not 0.0 < args.event_iou_threshold <= 1.0:
        raise ValueError("event-iou-threshold must lie in (0,1]")
    if args.boundary_dilation_frames != 0:
        raise ValueError(
            "derived activity-edge supervision requires boundary-dilation-frames=0; "
            "multi-frame positive boundaries are structurally contradictory"
        )
    if not 0.0 <= args.boundary_diagnostic_threshold <= 1.0:
        raise ValueError("boundary-diagnostic-threshold must lie in [0,1]")
    if args.onset_loss_weight < 0 or args.offset_loss_weight < 0:
        raise ValueError("boundary loss weights must be non-negative")
    if (args.max_train_scenes > 0 or args.max_dev_scenes > 0) and not args.allow_debug_data:
        raise RuntimeError("scene caps are debug-only and cannot produce a paper-eligible run")
    _seed_everything(args.seed)
    paths = {
        "train": args.train_manifest.resolve(),
        "dev": args.dev_manifest.resolve(),
        "test": args.test_manifest.resolve(),
    }
    ontology = args.ontology.resolve()
    initialization = args.initialization.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(ontology)
    gate = validate_gate_receipt(
        args.gate_receipt.resolve(),
        ontology=ontology,
        manifests=paths,
        allow_debug_data=args.allow_debug_data,
    )
    train_rows, train_raw = load_manifest(
        paths["train"], labels, expected_split="train", max_scenes=args.max_train_scenes
    )
    dev_rows, dev_raw = load_manifest(
        paths["dev"], labels, expected_split="dev", max_scenes=args.max_dev_scenes
    )
    calibration_rows, selection_rows = split_calibration_selection_rows(
        dev_rows,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    # Test is audited for leakage but never loaded as training/evaluation data.
    _, test_raw = load_manifest(paths["test"], labels, expected_split="test")
    identity_audit = audit_strict_identity_overlap(
        {"train": train_raw, "dev": dev_raw, "test": test_raw}
    )
    if not identity_audit["passes"]:
        raise RuntimeError(f"source identity overlaps across splits: {identity_audit['pairs']}")
    calibration_selection_identity = audit_strict_identity_overlap(
        {
            # The shared auditor enumerates canonical split keys only; map the
            # two dev subsets to train/dev aliases for this pairwise check.
            "train": [dict(row.raw) for row in calibration_rows],
            "dev": [dict(row.raw) for row in selection_rows],
        }
    )
    if not calibration_selection_identity["passes"]:
        raise RuntimeError(
            "source identity overlaps between calibration and selection: "
            f"{calibration_selection_identity['pairs']}"
        )

    train_coverage = class_coverage(train_rows, labels)
    dev_coverage = class_coverage(dev_rows, labels)
    calibration_coverage = class_coverage(calibration_rows, labels)
    selection_coverage = class_coverage(selection_rows, labels)
    missing_train = [label for label in labels if train_coverage[label]["scenes"] <= 0]
    if missing_train and not args.allow_debug_data:
        raise RuntimeError(f"train manifest has no positive scene for {len(missing_train)} labels")
    for split_name, coverage in (
        ("calibration", calibration_coverage),
        ("selection", selection_coverage),
    ):
        missing = [label for label in labels if coverage[label]["scenes"] <= 0]
        if missing and not args.allow_debug_data:
            raise RuntimeError(
                f"{split_name} split has no positive scene for {len(missing)} labels"
            )
    pos_weight, class_weight, sampler_weight = balancing_tensors(
        train_rows, labels, max_pos_weight=args.max_pos_weight
    )
    onset_pos_weight, offset_pos_weight = boundary_pos_weights(
        train_rows,
        num_labels=len(labels),
        dilation_frames=args.boundary_dilation_frames,
        max_pos_weight=args.max_boundary_pos_weight,
    )

    device = _device(args.device)
    payload = torch.load(initialization, map_location="cpu", weights_only=False)
    native_initialization_audit = validate_native_initialization(
        payload, labels=labels, ontology=ontology
    )
    model = load_model(200, "BEATs_strong_1", device, unfreeze_last_blocks=0)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if int(model.seq_len) != NUM_FRAMES:
        raise RuntimeError(f"expected BEATs seq_len=250, got {model.seq_len}")

    calibration_loader = _make_eval_loader(
        calibration_rows,
        audio_root=args.audio_root.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    selection_loader = _make_eval_loader(
        selection_rows,
        audio_root=args.audio_root.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    history: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0, -1.0, -1.0, -1.0)
    best_metrics: dict[str, Any] | None = None
    best_stage = "initialization"
    best_epoch = 0
    checkpoint_path = output_dir / "detector_best.pt"

    def evaluate_and_maybe_save(stage: str, epoch: int, loss: float | None) -> None:
        nonlocal best_score, best_metrics, best_stage, best_epoch
        calibration_predictions = collect_predictions(
            model, calibration_loader, device=device, amp=args.amp
        )
        calibration_metrics, curve = evaluate_thresholds(
            calibration_predictions,
            labels=labels,
            thresholds=args.thresholds,
            event_iou_threshold=args.event_iou_threshold,
            min_duration_seconds=args.min_event_duration,
            merge_gap_seconds=args.merge_gap,
            boundary_diagnostic_threshold=args.boundary_diagnostic_threshold,
        )
        selection_predictions = collect_predictions(
            model, selection_loader, device=device, amp=args.amp
        )
        selection_metrics = evaluate_fixed_grid_predictions(
            selection_predictions,
            labels=labels,
            threshold=float(calibration_metrics["threshold"]),
            event_iou_threshold=args.event_iou_threshold,
            min_duration_seconds=args.min_event_duration,
            merge_gap_seconds=args.merge_gap,
            boundary_diagnostic_threshold=args.boundary_diagnostic_threshold,
        )
        compact = _compact_epoch(
            stage=stage,
            epoch=epoch,
            loss=loss,
            calibration_metrics=calibration_metrics,
            selection_metrics=selection_metrics,
            curve=curve,
        )
        history.append(compact)
        score = _selection_score(selection_metrics)
        if score > best_score:
            best_score = score
            best_metrics = selection_metrics
            best_stage = stage
            best_epoch = epoch
            _atomic_torch(
                checkpoint_path,
                {
                    "format": CHECKPOINT_FORMAT,
                    "labels": labels,
                    "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "grid": {"frames": NUM_FRAMES, "hop_seconds": 0.04, "fixed_audio_seconds": 10.0},
                    "selected_threshold": calibration_metrics["threshold"],
                    "threshold_calibrated_on": "calibration_scenes_only",
                    "checkpoint_selected_on": "selection_scenes_only",
                    "best_dev": compact,
                    "initialization_sha256": _sha256(initialization),
                    "ontology_sha256": _sha256(ontology),
                },
            )
        running_receipt = {
            "format": FORMAT,
            "status": "running",
            "paper_eligible_data": bool(gate.get("paper_eligible")),
            "history": history,
            "best_stage": best_stage,
            "best_epoch": best_epoch,
        }
        _atomic_json(output_dir / "receipt.json", running_receipt)
        print(json.dumps(history[-1], ensure_ascii=False, sort_keys=True), flush=True)

    # Record the true native-selected baseline before any optimization.
    evaluate_and_maybe_save("initialization", 0, None)

    stages = [
        ("stage1_head_only", args.stage1_epochs, args.head_lr, 0.0),
        ("stage2_last_two_blocks", args.stage2_epochs, args.stage2_head_lr, args.stage2_backbone_lr),
    ]
    for stage, epochs, head_lr, backbone_lr in stages:
        if epochs <= 0:
            continue
        # Stage 2 starts from the best Stage 1/native checkpoint rather than
        # the last potentially overfit epoch.
        if stage == "stage2_last_two_blocks":
            best_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(best_payload["model_state_dict"], strict=True)
        head_params, backbone_params = configure_stage(model, stage)
        optimizer_groups: list[dict[str, Any]] = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            optimizer_groups.append({"params": backbone_params, "lr": backbone_lr})
        optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
        trainable = [*head_params, *backbone_params]
        for epoch in range(1, epochs + 1):
            train_loader = _make_train_loader(
                train_rows,
                audio_root=args.audio_root.resolve(),
                sampler_weights=sampler_weight,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                seed=args.seed + len(history) * 1009 + epoch,
            )
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                pos_weight=pos_weight.to(device),
                onset_pos_weight=onset_pos_weight.to(device),
                offset_pos_weight=offset_pos_weight.to(device),
                class_weight=class_weight.to(device),
                device=device,
                amp=args.amp,
                gradient_clip=args.gradient_clip,
                onset_loss_weight=args.onset_loss_weight,
                offset_loss_weight=args.offset_loss_weight,
                boundary_dilation_frames=args.boundary_dilation_frames,
                trainable_parameters=trainable,
            )
            evaluate_and_maybe_save(stage, epoch, loss)

    if best_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("training produced no valid checkpoint")
    receipt = {
        "format": FORMAT,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_eligible_data": bool(gate.get("paper_eligible")),
        "data_gate": {
            "path": str(args.gate_receipt.resolve()),
            "sha256": _sha256(args.gate_receipt.resolve()) if args.gate_receipt.is_file() else None,
            "format": gate.get("format"),
        },
        "inputs": {
            "ontology": {"path": str(ontology), "sha256": _sha256(ontology), "labels": len(labels)},
            "initialization": {"path": str(initialization), "sha256": _sha256(initialization)},
            "manifests": {
                split: {"path": str(path), "sha256": _sha256(path)} for split, path in paths.items()
            },
        },
        "integrity": identity_audit,
        "calibration_selection_identity_audit": calibration_selection_identity,
        "native_initialization_audit": native_initialization_audit,
        "grid": {
            "frames": NUM_FRAMES,
            "hop_seconds": 0.04,
            "fixed_audio_seconds": 10.0,
            "right_padding_excluded_from_loss_and_metrics": True,
            "timestamp_projection_uses_clip_duration": False,
        },
        "training": {
            "stage1": {"epochs": args.stage1_epochs, "trainable": "strong_head_only"},
            "stage2": {"epochs": args.stage2_epochs, "trainable": "strong_head+BEATs_blocks_10_11"},
            "class_balanced_sampler": True,
            "class_balanced_bce": True,
            "onset_supervision": {
                "weight": args.onset_loss_weight,
                "dilation_frames": args.boundary_dilation_frames,
                "derived_probability": "activity[t] * (1-activity[t-1])",
            },
            "offset_supervision": {
                "weight": args.offset_loss_weight,
                "dilation_frames": args.boundary_dilation_frames,
                "derived_probability": "activity[t-1] * (1-activity[t])",
            },
            "boundary_metric": {
                "threshold": args.boundary_diagnostic_threshold,
                "diagnostic_only": True,
                "used_for_threshold_calibration": False,
                "used_for_checkpoint_selection": False,
            },
            "threshold_calibration": {
                "calibration_fraction": args.calibration_fraction,
                "calibration_scene_count": len(calibration_rows),
                "selection_scene_count": len(selection_rows),
                "calibrated_on_selection": False,
                "test_used": False,
            },
            "seed": args.seed,
        },
        "coverage": {
            "train": train_coverage,
            "dev_full": dev_coverage,
            "calibration": calibration_coverage,
            "selection": selection_coverage,
        },
        "selection_metric": "selection event macro-F1, event micro-F1, frame macro/micro; oracle-window only final tie-break",
        "best_stage": best_stage,
        "best_epoch": best_epoch,
        "best_dev": best_metrics,
        "history": history,
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
    }
    _atomic_json(output_dir / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    main()
