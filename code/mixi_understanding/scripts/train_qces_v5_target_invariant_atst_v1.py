#!/usr/bin/env python3
"""Target-invariant ATST adaptation for the frozen QCES 188-class contract.

This is one locked representation experiment, not a hyper-parameter sweep:

* accepted Gold train sources only, with dev/matched source and video blocking;
* one clean view and two dynamic hard-negative remixes per target;
* the final two ATST blocks plus the frame norm are trainable;
* the already validated seven expert heads are warm-started and retained;
* a shared projector learns group routing and clean/remix invariance;
* development selects the checkpoint; matched evaluation is opened once.

The script reports both deployable predicted-router accuracy and oracle-group
accuracy.  Oracle-group is a diagnostic ceiling and is never called deployable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
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
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch import nn
from torch.utils.data import DataLoader, Dataset

import models.prediction_wrapper as prediction_wrapper_module
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    SceneDataset,
    SceneItem,
    collate,
    load_ontology,
    load_scene_manifest,
    make_device,
    set_seed,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _atomic_torch,
    _sha256_file,
)
from mixi_understanding.scripts.train_qces_v5_atst_oracle_semantic_screen_v1 import (
    CachedSceneDataset,
    cached_collate,
    collect_spans,
)
from mixi_understanding.scripts.audit_qces_v5_matched_clean_mixture_gap_v1 import (
    load_component_canvas,
)
from mixi_understanding.scripts.train_qces_v5_oracle_group_experts_v1 import (
    GroupExperts,
    expert_ordering,
    group_lookup,
    loader as cached_loader,
    ordering_metrics,
)


FORMAT = "qces_v5_target_invariant_atst_v1"
SAMPLE_RATE = 16_000
FIXED_SECONDS = 10.0
FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_SECONDS)
NUM_FRAMES = 250
OVERLAP_CHOICES = (0.25, 0.50, 0.75, 1.00)


def parse_args() -> argparse.Namespace:
    data = Path("/var/tmp/qces_full188_tiered_realistic_v5")
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    previous = base / "v5_oracle_group_experts_v1/oracle_group_experts_v1_best.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-index", type=Path, default=PROJECT_ROOT / "data_full/index/accepted_samples.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--dev-manifest", type=Path, default=data / "detector_scene_manifest_tiered_dev.jsonl")
    parser.add_argument("--matched-manifest", type=Path, default=Path("/var/tmp/qces_v5_matched_eval_v1/detector_scene_manifest_matched_eval.jsonl"))
    parser.add_argument("--previous-experts", type=Path, default=previous)
    parser.add_argument("--previous-matched-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_matched_eval_v1/atst_features_matched_eval.pt"))
    parser.add_argument("--source-pool", type=Path, default=Path("/var/tmp/qces_v5_target_invariant_atst_v1/gold_train_source_pool.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=base / "v5_target_invariant_atst_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12031)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=3760)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--group-loss-weight", type=float, default=0.20)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.20)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--sir-min-db", type=float, default=-10.0)
    parser.add_argument("--sir-max-db", type=float, default=10.0)
    parser.add_argument("--max-active-seconds", type=float, default=6.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--skip-matched", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def resolved_audio_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def blocked_eval_identities(paths: Sequence[Path]) -> dict[str, set[str]]:
    blocked: dict[str, set[str]] = {
        "source_id": set(),
        "video_id": set(),
        "source_path": set(),
        "source_sha256": set(),
    }
    for path in paths:
        for scene in read_jsonl(path):
            for event in scene.get("events", []):
                if event.get("source_id"):
                    blocked["source_id"].add(str(event["source_id"]))
                if event.get("source_video_id"):
                    blocked["video_id"].add(str(event["source_video_id"]))
                if event.get("source_path"):
                    blocked["source_path"].add(resolved_audio_path(str(event["source_path"])).as_posix())
                if event.get("source_sha256"):
                    blocked["source_sha256"].add(str(event["source_sha256"]))
    return blocked


def build_source_pool(
    accepted_index: Path,
    labels: Sequence[str],
    blocked: Mapping[str, set[str]],
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    selected: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    for raw in read_jsonl(accepted_index):
        label = str(raw.get("label") or "")
        if label not in label_to_id:
            rejected["outside_frozen_ontology"] += 1
            continue
        if str(raw.get("split") or "").lower() != "train":
            rejected["not_train"] += 1
            continue
        if str(raw.get("quality_tier") or "").lower() != "gold":
            rejected["not_gold"] += 1
            continue
        source_id = str(raw.get("sample_id") or "")
        video_id = str(raw.get("video_id") or "")
        path = resolved_audio_path(str(raw.get("audio_path") or ""))
        source_sha = str(raw.get("audio_sha256") or raw.get("source_sha256") or "")
        if not source_id or source_id in seen:
            rejected["missing_or_duplicate_source"] += 1
            continue
        if source_id in blocked["source_id"]:
            rejected["eval_source_id"] += 1
            continue
        if video_id and video_id in blocked["video_id"]:
            rejected["eval_video_id"] += 1
            continue
        if path.as_posix() in blocked["source_path"]:
            rejected["eval_source_path"] += 1
            continue
        if source_sha and source_sha in blocked["source_sha256"]:
            rejected["eval_source_sha256"] += 1
            continue
        if not path.is_file():
            rejected["missing_audio"] += 1
            continue
        onset = max(0.0, float(raw.get("active_onset_seconds") or 0.0))
        duration = max(0.0, float(raw.get("duration_seconds") or 0.0))
        offset = float(raw.get("active_offset_seconds") or duration)
        if offset <= onset + 0.04:
            rejected["invalid_active_interval"] += 1
            continue
        row = {
            "sample_id": source_id,
            "video_id": video_id,
            "label": label,
            "label_id": label_to_id[label],
            "audio_path": path.as_posix(),
            "active_onset_seconds": onset,
            "active_offset_seconds": offset,
            "quality_tier": "gold",
            "source_dataset": str(raw.get("source_dataset") or ""),
        }
        selected.append(row)
        seen.add(source_id)
    selected.sort(key=lambda row: (int(row["label_id"]), str(row["sample_id"])))
    counts = Counter(int(row["label_id"]) for row in selected)
    missing = [labels[index] for index in range(len(labels)) if counts[index] == 0]
    source_overlap = {str(row["sample_id"]) for row in selected} & blocked["source_id"]
    video_overlap = ({str(row["video_id"]) for row in selected} - {""}) & blocked["video_id"]
    path_overlap = {str(row["audio_path"]) for row in selected} & blocked["source_path"]
    audit = {
        "selected_sources": len(selected),
        "observed_classes": len(counts),
        "missing_classes": missing,
        "minimum_per_class": min(counts.values()) if counts else 0,
        "median_per_class": float(np.median(list(counts.values()))) if counts else 0.0,
        "maximum_per_class": max(counts.values()) if counts else 0,
        "source_id_overlap_with_dev_matched": len(source_overlap),
        "video_id_overlap_with_dev_matched": len(video_overlap),
        "source_path_overlap_with_dev_matched": len(path_overlap),
        "rejections": dict(rejected),
    }
    gates = {
        "exactly_188_observed_classes": len(counts) == 188,
        # Twelve is the locked independence floor.  The observed minimum is 14
        # (Whispering) after removing every dev/matched identity; demanding 15
        # would either leak evaluation sources or silently remove a frozen class.
        "minimum_12_gold_sources_per_class": bool(counts) and min(counts.values()) >= 12,
        "zero_source_id_leakage": not source_overlap,
        "zero_video_id_leakage": not video_overlap,
        "zero_source_path_leakage": not path_overlap,
    }
    audit["gates"] = {"passed": all(gates.values()), "checks": gates}
    if not audit["gates"]["passed"]:
        raise RuntimeError(f"source-pool gate failed: {audit}")
    atomic_jsonl(selected, output_path)
    return selected, audit


@dataclass(frozen=True)
class SourceRow:
    sample_id: str
    video_id: str
    label: str
    label_id: int
    audio_path: str
    active_onset_seconds: float
    active_offset_seconds: float


class DynamicRemixDataset(Dataset[dict[str, Any]]):
    """Balanced target sampling with one clean and two hard-remix views."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        groups: Sequence[Sequence[int]],
        *,
        samples_per_epoch: int,
        seed: int,
        sir_min_db: float,
        sir_max_db: float,
        max_active_seconds: float,
    ) -> None:
        self.rows = [SourceRow(**{key: row[key] for key in SourceRow.__dataclass_fields__}) for row in rows]
        self.groups = [list(map(int, group)) for group in groups]
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.sir_min_db = float(sir_min_db)
        self.sir_max_db = float(sir_max_db)
        self.max_active_samples = int(round(float(max_active_seconds) * SAMPLE_RATE))
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.by_label[int(row.label_id)].append(index)
        self.labels = sorted(self.by_label)
        self.label_to_group = {}
        for group_id, group in enumerate(self.groups):
            for label_id in group:
                self.label_to_group[label_id] = group_id
        if self.labels != list(range(188)):
            raise ValueError("dynamic source pool does not cover the exact 188-class contract")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    @staticmethod
    def _load(row: SourceRow) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(row.audio_path)
        waveform = waveform.float().mean(0) if waveform.ndim == 2 else waveform.float().reshape(-1)
        if sample_rate != SAMPLE_RATE:
            waveform = AF.resample(waveform, sample_rate, SAMPLE_RATE)
        start = max(0, min(waveform.numel() - 1, int(math.floor(row.active_onset_seconds * SAMPLE_RATE))))
        end = max(start + 1, min(waveform.numel(), int(math.ceil(row.active_offset_seconds * SAMPLE_RATE))))
        return waveform[start:end]

    @staticmethod
    def _maximum_energy_crop(value: torch.Tensor, maximum: int) -> torch.Tensor:
        if value.numel() <= maximum:
            return value
        squared = value.double().square()
        prefix = F.pad(squared.cumsum(0), (1, 0))
        energy = prefix[maximum:] - prefix[:-maximum]
        start = int(energy.argmax())
        return value[start : start + maximum]

    def _segment(self, row: SourceRow) -> torch.Tensor:
        value = self._maximum_energy_crop(self._load(row), self.max_active_samples)
        if value.numel() < int(0.04 * SAMPLE_RATE):
            value = F.pad(value, (0, int(0.04 * SAMPLE_RATE) - value.numel()))
        rms = value.square().mean().sqrt().clamp_min(1e-5)
        value = value * (0.08 / rms)
        peak = value.abs().amax().clamp_min(1e-6)
        if float(peak) > 0.95:
            value = value * (0.95 / peak)
        return value

    @staticmethod
    def _place(value: torch.Tensor, onset: int) -> torch.Tensor:
        canvas = value.new_zeros(FIXED_SAMPLES)
        stop = min(FIXED_SAMPLES, onset + value.numel())
        canvas[onset:stop] = value[: stop - onset]
        return canvas

    @staticmethod
    def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
        return max(0, min(a_end, b_end) - max(a_start, b_start))

    def _distractor_onset(
        self,
        target_start: int,
        target_length: int,
        distractor_length: int,
        overlap_fraction: float,
        rng: random.Random,
    ) -> int:
        desired = int(round(overlap_fraction * min(target_length, distractor_length)))
        candidates = [
            target_start + target_length - desired,
            target_start - distractor_length + desired,
        ]
        candidates = [max(0, min(FIXED_SAMPLES - distractor_length, value)) for value in candidates]
        rng.shuffle(candidates)
        return min(
            candidates,
            key=lambda value: abs(
                self._overlap(
                    target_start,
                    target_start + target_length,
                    value,
                    value + distractor_length,
                )
                - desired
            ),
        )

    def _mix_one(
        self,
        target: torch.Tensor,
        target_start: int,
        distractors: Sequence[SourceRow],
        rng: random.Random,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        clean = self._place(target, target_start)
        mixture = clean.clone()
        records: list[dict[str, Any]] = []
        for row in distractors:
            segment = self._segment(row)
            overlap_fraction = rng.choice(OVERLAP_CHOICES)
            onset = self._distractor_onset(
                target_start, target.numel(), segment.numel(), overlap_fraction, rng
            )
            canvas = self._place(segment, onset)
            overlap_start = max(target_start, onset)
            overlap_end = min(target_start + target.numel(), onset + segment.numel())
            if overlap_end > overlap_start:
                target_rms = clean[overlap_start:overlap_end].square().mean().sqrt().clamp_min(1e-5)
                distractor_rms = canvas[overlap_start:overlap_end].square().mean().sqrt().clamp_min(1e-5)
            else:
                target_rms = target.square().mean().sqrt().clamp_min(1e-5)
                distractor_rms = segment.square().mean().sqrt().clamp_min(1e-5)
            desired_sir = rng.uniform(self.sir_min_db, self.sir_max_db)
            gain = target_rms / (distractor_rms * (10.0 ** (desired_sir / 20.0)))
            scaled = canvas * gain
            mixture.add_(scaled)
            actual_overlap = self._overlap(
                target_start,
                target_start + target.numel(),
                onset,
                onset + segment.numel(),
            )
            records.append(
                {
                    "label_id": int(row.label_id),
                    "desired_sir_db": float(desired_sir),
                    "requested_overlap_fraction": float(overlap_fraction),
                    "actual_overlap_fraction": actual_overlap / max(min(target.numel(), segment.numel()), 1),
                }
            )
        peak = mixture.abs().amax().clamp_min(1e-6)
        global_scale = min(1.0, 0.98 / float(peak))
        mixture.mul_(global_scale)
        return mixture, {"distractors": records, "global_scale": global_scale}

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + int(index) * 104_729)
        label_id = int((index + self.epoch * 37) % len(self.labels))
        row = self.rows[rng.choice(self.by_label[label_id])]
        target = self._segment(row)
        target_start = rng.randint(0, FIXED_SAMPLES - target.numel())
        clean = self._place(target, target_start)
        group_id = int(self.label_to_group[label_id])
        hard_labels = [value for value in self.groups[group_id] if value != label_id and self.by_label[value]]
        hard_one = self.rows[rng.choice(self.by_label[rng.choice(hard_labels)])]
        hard_two = self.rows[rng.choice(self.by_label[rng.choice(hard_labels)])]
        other_labels = [value for value in self.labels if value != label_id and value not in self.groups[group_id]]
        random_negative = self.rows[rng.choice(self.by_label[rng.choice(other_labels)])]
        mix_one, meta_one = self._mix_one(target, target_start, [hard_one], rng)
        mix_two, meta_two = self._mix_one(target, target_start, [hard_two, random_negative], rng)
        return {
            "views": torch.stack((clean, mix_one, mix_two)),
            "interval": torch.tensor(
                [target_start / FIXED_SAMPLES, (target_start + target.numel()) / FIXED_SAMPLES],
                dtype=torch.float32,
            ),
            "label": label_id,
            "group": group_id,
            "sample_id": row.sample_id,
            "meta": [meta_one, meta_two],
        }


def remix_collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "views": torch.stack([row["views"] for row in rows]),
        "interval": torch.stack([row["interval"] for row in rows]),
        "label": torch.tensor([int(row["label"]) for row in rows], dtype=torch.long),
        "group": torch.tensor([int(row["group"]) for row in rows], dtype=torch.long),
        "sample_id": [str(row["sample_id"]) for row in rows],
        "meta": [row["meta"] for row in rows],
    }


class SharedInvariantProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_groups: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.group_head = nn.Linear(hidden_dim, num_groups)

    def forward(self, spans: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = mask.unsqueeze(-1).to(spans.dtype)
        denominator = weight.sum(1).clamp_min(1.0)
        mean = (spans * weight).sum(1) / denominator
        maximum = spans.masked_fill(~mask.unsqueeze(-1), torch.finfo(spans.dtype).min).amax(1)
        variance = ((spans - mean[:, None]) ** 2 * weight).sum(1) / denominator
        embedding = self.projector(torch.cat((mean, maximum, variance.sqrt()), dim=-1))
        embedding = F.normalize(embedding, dim=-1)
        return embedding, self.group_head(embedding)


class TargetInvariantExperts(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        groups: Sequence[Sequence[int]],
        previous_state: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.experts = GroupExperts(input_dim, hidden_dim, groups)
        self.experts.load_state_dict(previous_state, strict=True)
        self.shared = SharedInvariantProjector(input_dim, hidden_dim, len(groups))


def build_backbone(num_classes: int, device: torch.device) -> PredictionsWrapper:
    prediction_wrapper_module.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    backbone = PredictionsWrapper(
        ATSTWrapper(),
        checkpoint="ATST-F_strong_1",
        n_classes_strong=num_classes,
        n_classes_weak=num_classes,
        seq_model_type=None,
        head_type="linear",
    ).to(device)
    backbone.requires_grad_(False)
    for name, parameter in backbone.model.named_parameters():
        if "atst.blocks.10." in name or "atst.blocks.11." in name or "atst.norm_frame." in name:
            parameter.requires_grad = True
    trainable = [name for name, value in backbone.model.named_parameters() if value.requires_grad]
    if not trainable or any(
        not ("atst.blocks.10." in name or "atst.blocks.11." in name or "atst.norm_frame." in name)
        for name in trainable
    ):
        raise RuntimeError(f"unexpected ATST trainable parameter contract: {trainable[:10]}")
    return backbone


def backbone_features(
    backbone: PredictionsWrapper,
    audio: torch.Tensor,
    *,
    amp: bool,
) -> torch.Tensor:
    device_type = audio.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=bool(amp and device_type == "cuda"),
    ):
        mel = backbone.mel_forward(audio.float())
        value = backbone.model(mel)
    if value.shape[1] != NUM_FRAMES:
        value = F.interpolate(
            value.transpose(1, 2), size=NUM_FRAMES, mode="linear", align_corners=False
        ).transpose(1, 2)
    return value.float()


def interval_spans(features: torch.Tensor, intervals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sequences: list[torch.Tensor] = []
    for feature, interval in zip(features, intervals, strict=True):
        start = max(0, min(NUM_FRAMES - 1, int(math.floor(float(interval[0]) * NUM_FRAMES))))
        end = max(start + 1, min(NUM_FRAMES, int(math.ceil(float(interval[1]) * NUM_FRAMES))))
        sequences.append(feature[start:end])
    maximum = max(value.shape[0] for value in sequences)
    padded = features.new_zeros((len(sequences), maximum, features.shape[-1]))
    mask = torch.zeros((len(sequences), maximum), dtype=torch.bool, device=features.device)
    for index, value in enumerate(sequences):
        padded[index, : value.shape[0]] = value
        mask[index, : value.shape[0]] = True
    return padded, mask


def local_lookups(groups: Sequence[Sequence[int]], num_classes: int, device: torch.device) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    for group in groups:
        mapping = {int(label): index for index, label in enumerate(group)}
        result.append(torch.tensor([mapping.get(label, -1) for label in range(num_classes)], device=device))
    return result


def fine_classification_loss(
    experts: GroupExperts,
    spans: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    target_groups: torch.Tensor,
    groups: Sequence[Sequence[int]],
    lookups: Sequence[torch.Tensor],
    label_smoothing: float,
) -> tuple[torch.Tensor, int, int]:
    summed = spans.new_zeros(())
    correct = seen = 0
    for group_id, head in enumerate(experts.heads):
        selected = torch.nonzero(target_groups == group_id, as_tuple=False).flatten()
        if not selected.numel():
            continue
        local_target = lookups[group_id][targets[selected]]
        logits = head(spans[selected], mask[selected])
        summed = summed + F.cross_entropy(
            logits,
            local_target,
            label_smoothing=label_smoothing,
            reduction="sum",
        )
        correct += int(logits.argmax(1).eq(local_target).sum())
        seen += int(selected.numel())
    return summed / max(seen, 1), correct, seen


def nt_xent(embedding: torch.Tensor, sample_index: torch.Tensor, temperature: float) -> torch.Tensor:
    if sample_index.unique().numel() < 2:
        return embedding.new_zeros(())
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = sample_index[:, None].eq(sample_index[None, :]) & ~identity
    valid = ~identity
    denominator = torch.logsumexp(similarity.masked_fill(~valid, -torch.inf), dim=1)
    numerator = torch.logsumexp(similarity.masked_fill(~positive, -torch.inf), dim=1)
    return (denominator - numerator).mean()


def classify_orders(
    model: TargetInvariantExperts,
    spans: torch.Tensor,
    mask: torch.Tensor,
    target_groups: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding, group_logits = model.shared(spans, mask)
    predicted_groups = group_logits.argmax(1)
    maximum = max(map(len, groups))
    oracle_order = torch.full((len(spans), maximum), -1, dtype=torch.long, device=spans.device)
    routed_order = torch.full_like(oracle_order, -1)
    for group_id, (head, group) in enumerate(zip(model.experts.heads, groups, strict=True)):
        global_labels = torch.tensor(group, dtype=torch.long, device=spans.device)
        oracle_selected = torch.nonzero(target_groups == group_id, as_tuple=False).flatten()
        if oracle_selected.numel():
            local = head(spans[oracle_selected], mask[oracle_selected]).argsort(1, descending=True)
            oracle_order[oracle_selected, : len(group)] = global_labels[local]
        routed_selected = torch.nonzero(predicted_groups == group_id, as_tuple=False).flatten()
        if routed_selected.numel():
            local = head(spans[routed_selected], mask[routed_selected]).argsort(1, descending=True)
            routed_order[routed_selected, : len(group)] = global_labels[local]
    return oracle_order, routed_order, predicted_groups


def semantic_events(row: SceneItem) -> list[Mapping[str, Any]]:
    return [event for event in row.events if str(event.get("event_kind", "semantic")) == "semantic"]


class ComponentCanvasDataset(Dataset[tuple[torch.Tensor, SceneItem]]):
    """Reinsert compact clean components at their mixture-aligned onset."""

    def __init__(self, rows: Sequence[SceneItem]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, SceneItem]:
        row = self.rows[index]
        if len(row.events) != 1:
            raise ValueError("clean component rows must contain exactly one event")
        return load_component_canvas(dict(row.events[0])), row


@torch.inference_mode()
def evaluate_rows(
    backbone: PredictionsWrapper,
    model: TargetInvariantExperts,
    rows: Sequence[SceneItem],
    groups: Sequence[Sequence[int]],
    label_to_group: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    amp: bool,
    component_canvas: bool = False,
) -> dict[str, Any]:
    dataset: Dataset[tuple[torch.Tensor, SceneItem]]
    if component_canvas:
        dataset = ComponentCanvasDataset(rows)
    else:
        dataset = SceneDataset(rows, audio_root=Path("/"), fixed_seconds=FIXED_SECONDS)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )
    backbone.eval(); model.eval()
    oracle_rows: list[torch.Tensor] = []
    routed_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    predicted_group_rows: list[torch.Tensor] = []
    overlap_rows: list[float] = []
    processed = 0
    for audio, batch_rows in data_loader:
        features = backbone_features(backbone, audio.to(device, non_blocking=True), amp=amp)
        intervals: list[torch.Tensor] = []
        targets: list[int] = []
        for batch_index, row in enumerate(batch_rows):
            for event in semantic_events(row):
                intervals.append(
                    torch.tensor(
                        [
                            batch_index,
                            float(event["onset_seconds"]) / FIXED_SECONDS,
                            float(event["offset_seconds"]) / FIXED_SECONDS,
                        ],
                        device=device,
                    )
                )
                targets.append(int(event["label_id"]))
                overlap_rows.append(float(event.get("overlap_fraction") or 0.0))
        sequences: list[torch.Tensor] = []
        for batch_index, start_fraction, end_fraction in torch.stack(intervals):
            start = max(0, min(NUM_FRAMES - 1, int(math.floor(float(start_fraction) * NUM_FRAMES))))
            end = max(start + 1, min(NUM_FRAMES, int(math.ceil(float(end_fraction) * NUM_FRAMES))))
            sequences.append(features[int(batch_index), start:end])
        maximum = max(value.shape[0] for value in sequences)
        spans = features.new_zeros((len(sequences), maximum, features.shape[-1]))
        mask = torch.zeros((len(sequences), maximum), dtype=torch.bool, device=device)
        for index, value in enumerate(sequences):
            spans[index, : value.shape[0]] = value
            mask[index, : value.shape[0]] = True
        target = torch.tensor(targets, dtype=torch.long, device=device)
        target_group = label_to_group.to(device)[target]
        oracle, routed, predicted_group = classify_orders(model, spans, mask, target_group, groups)
        oracle_rows.append(oracle.cpu())
        routed_rows.append(routed.cpu())
        target_rows.append(target.cpu())
        predicted_group_rows.append(predicted_group.cpu())
        processed += len(batch_rows)
    oracle = torch.cat(oracle_rows)
    routed = torch.cat(routed_rows)
    target = torch.cat(target_rows)
    predicted_group = torch.cat(predicted_group_rows)
    target_group = label_to_group[target]
    overlap = torch.tensor(overlap_rows)
    heavy = overlap >= 0.30
    return {
        "oracle_order": oracle,
        "routed_order": routed,
        "targets": target,
        "predicted_groups": predicted_group,
        "oracle_group": ordering_metrics(oracle, target),
        "predicted_router": ordering_metrics(routed, target),
        "router_group_accuracy_↑": float(predicted_group.eq(target_group).float().mean()),
        "heavy_overlap_ge_0_30": ordering_metrics(routed[heavy], target[heavy]) if heavy.any() else None,
        "heavy_overlap_oracle_group": ordering_metrics(oracle[heavy], target[heavy]) if heavy.any() else None,
        "heavy_mask": heavy,
    }


def component_rows(rows: Sequence[SceneItem]) -> list[SceneItem]:
    result: list[SceneItem] = []
    for row in rows:
        for event in semantic_events(row):
            result.append(
                SceneItem(
                    scene_id=str(event.get("event_id") or f"{row.scene_id}:{len(result)}"),
                    split=row.split + "_clean_component",
                    mixture_path=str(event["component_path"]),
                    duration_seconds=FIXED_SECONDS,
                    sample_rate=SAMPLE_RATE,
                    events=(dict(event),),
                )
            )
    return result


def old_expert_baseline(
    checkpoint_path: Path,
    cache_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    groups = [list(map(int, value)) for value in payload["groups"]]
    label_to_group, _ = group_lookup(groups, len(payload["labels"]))
    old = GroupExperts(int(payload["input_dim"]), int(payload["hidden_dim"]), groups).to(device)
    old.load_state_dict(payload["model_state_dict"], strict=True)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    data_loader = cached_loader(cache, 16, shuffle=False, seed=0)
    ordering, targets = expert_ordering(old, data_loader, device, groups, label_to_group)
    del old
    return {"ordering": ordering, "targets": targets, "metrics": ordering_metrics(ordering, targets)}


def sha256_text(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8")); digest.update(b"\n")
    return digest.hexdigest()


def stripped_eval(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "oracle_group": result["oracle_group"],
        "predicted_router": result["predicted_router"],
        "router_group_accuracy_↑": result["router_group_accuracy_↑"],
        "heavy_overlap_ge_0_30": result["heavy_overlap_ge_0_30"],
        "heavy_overlap_oracle_group": result["heavy_overlap_oracle_group"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_ontology(args.ontology.resolve())
    if len(labels) != 188:
        raise ValueError(f"expected frozen 188-class ontology, got {len(labels)}")
    label_to_id = {label: index for index, label in enumerate(labels)}
    previous_path = args.previous_experts.resolve()
    previous = torch.load(previous_path, map_location="cpu", weights_only=False)
    if list(previous["labels"]) != labels:
        raise ValueError("previous expert ontology/order differs from frozen ontology")
    groups = [list(map(int, group)) for group in previous["groups"]]
    if len(groups) != 7 or sorted(value for group in groups for value in group) != list(range(188)):
        raise ValueError("previous seven-group partition is invalid")
    label_to_group, _ = group_lookup(groups, len(labels))

    blocked = blocked_eval_identities([args.dev_manifest.resolve(), args.matched_manifest.resolve()])
    source_rows, source_audit = build_source_pool(
        args.accepted_index.resolve(), labels, blocked, args.source_pool.resolve()
    )
    print(json.dumps({"source_pool": source_audit}, sort_keys=True), flush=True)

    dataset = DynamicRemixDataset(
        source_rows,
        groups,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        sir_min_db=args.sir_min_db,
        sir_max_db=args.sir_max_db,
        max_active_seconds=args.max_active_seconds,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=remix_collate,
        persistent_workers=False,
        generator=loader_generator,
    )
    # Deterministic waveform contract audit before any optimizer step.
    preview = [dataset[index] for index in range(min(16, len(dataset)))]
    overlap_values = [
        item["actual_overlap_fraction"]
        for row in preview for view in row["meta"] for item in view["distractors"]
    ]
    sir_values = [
        item["desired_sir_db"]
        for row in preview for view in row["meta"] for item in view["distractors"]
    ]
    remix_audit = {
        "audited_targets": len(preview),
        "all_views_shape_3x160000": all(tuple(row["views"].shape) == (3, FIXED_SAMPLES) for row in preview),
        "all_waveforms_finite": all(bool(torch.isfinite(row["views"]).all()) for row in preview),
        "all_target_intervals_valid": all(0 <= float(row["interval"][0]) < float(row["interval"][1]) <= 1 for row in preview),
        "all_mixtures_differ_from_clean": all(
            not torch.equal(row["views"][0], row["views"][1]) and not torch.equal(row["views"][0], row["views"][2])
            for row in preview
        ),
        "minimum_actual_overlap_fraction": min(overlap_values),
        "maximum_actual_overlap_fraction": max(overlap_values),
        "minimum_requested_sir_db": min(sir_values),
        "maximum_requested_sir_db": max(sir_values),
    }
    if not all(value for key, value in remix_audit.items() if key.startswith("all_")):
        raise RuntimeError(f"dynamic remix waveform contract failed: {remix_audit}")
    print(json.dumps({"remix_audit": remix_audit}, sort_keys=True), flush=True)

    device = make_device(args.device)
    backbone = build_backbone(len(labels), device)
    model = TargetInvariantExperts(
        int(previous["input_dim"]), int(previous["hidden_dim"]), groups, previous["model_state_dict"]
    ).to(device)
    trainable_backbone_names = [name for name, value in backbone.model.named_parameters() if value.requires_grad]
    trainable_backbone_parameters = [value for value in backbone.model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": model.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.backbone_learning_rate * 0.05
    )
    lookups = local_lookups(groups, len(labels), device)
    device_label_to_group = label_to_group.to(device)

    dev_rows = load_scene_manifest(args.dev_manifest.resolve(), label_to_id)
    if args.max_dev_scenes:
        dev_rows = dev_rows[: args.max_dev_scenes]
    best_key: tuple[float, float, float, float] | None = None
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_backbone_state: dict[str, torch.Tensor] | None = None
    best_dev: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    gradient_audit: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        backbone.train(); model.train()
        loss_sum = fine_sum = group_sum = contrast_sum = consistency_sum = 0.0
        correct = seen = 0
        processed_batches = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch_size, views, samples = batch["views"].shape
            audio = batch["views"].reshape(batch_size * views, samples).to(device, non_blocking=True)
            interval = batch["interval"].to(device).repeat_interleave(views, dim=0)
            targets = batch["label"].to(device).repeat_interleave(views)
            target_groups = batch["group"].to(device).repeat_interleave(views)
            features = backbone_features(backbone, audio, amp=args.amp)
            spans, mask = interval_spans(features, interval)
            fine_loss, batch_correct, batch_seen = fine_classification_loss(
                model.experts,
                spans,
                mask,
                targets,
                target_groups,
                groups,
                lookups,
                args.label_smoothing,
            )
            embedding, group_logits = model.shared(spans, mask)
            group_loss = F.cross_entropy(group_logits, target_groups, label_smoothing=0.02)
            sample_index = torch.arange(batch_size, device=device).repeat_interleave(views)
            contrastive = nt_xent(embedding, sample_index, args.contrastive_temperature)
            shaped_embedding = embedding.reshape(batch_size, views, -1)
            consistency = (1.0 - (shaped_embedding[:, :1] * shaped_embedding[:, 1:]).sum(-1)).mean()
            loss = (
                fine_loss
                + args.group_loss_weight * group_loss
                + args.contrastive_loss_weight * contrastive
                + args.consistency_loss_weight * consistency
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_audit is None:
                trainable_gradient = {
                    name: float(parameter.grad.detach().norm()) if parameter.grad is not None else 0.0
                    for name, parameter in backbone.model.named_parameters()
                    if parameter.requires_grad
                }
                frozen_with_gradient = [
                    name for name, parameter in backbone.model.named_parameters()
                    if not parameter.requires_grad and parameter.grad is not None
                ]
                gradient_audit = {
                    "trainable_parameter_tensors": len(trainable_gradient),
                    "trainable_gradient_norm_sum": sum(trainable_gradient.values()),
                    "trainable_zero_gradient_tensors": sum(value == 0.0 for value in trainable_gradient.values()),
                    "frozen_parameters_with_gradient": frozen_with_gradient,
                    "only_blocks_10_11_and_norm_trainable": all(
                        "atst.blocks.10." in name or "atst.blocks.11." in name or "atst.norm_frame." in name
                        for name in trainable_gradient
                    ),
                }
                if (
                    gradient_audit["trainable_gradient_norm_sum"] <= 0
                    or gradient_audit["frozen_parameters_with_gradient"]
                    or not gradient_audit["only_blocks_10_11_and_norm_trainable"]
                ):
                    raise RuntimeError(f"gradient contract failed: {gradient_audit}")
                print(json.dumps({"gradient_audit": gradient_audit}, sort_keys=True), flush=True)
            torch.nn.utils.clip_grad_norm_(trainable_backbone_parameters, 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * batch_seen
            fine_sum += float(fine_loss.detach()) * batch_seen
            group_sum += float(group_loss.detach()) * batch_seen
            contrast_sum += float(contrastive.detach()) * batch_seen
            consistency_sum += float(consistency.detach()) * batch_seen
            correct += batch_correct; seen += batch_seen; processed_batches += 1
            if batch_index == 1 or batch_index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "train",
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches_total": math.ceil(len(dataset) / args.batch_size),
                            "running_fine_top1": correct / max(seen, 1),
                            "running_loss": loss_sum / max(seen, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        scheduler.step()
        dev = evaluate_rows(
            backbone,
            model,
            dev_rows,
            groups,
            label_to_group,
            device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
        )
        row = {
            "epoch": epoch,
            "train": {
                "views": seen,
                "fine_top1_↑": correct / max(seen, 1),
                "loss": loss_sum / max(seen, 1),
                "fine_loss": fine_sum / max(seen, 1),
                "group_loss": group_sum / max(seen, 1),
                "contrastive_loss": contrast_sum / max(seen, 1),
                "consistency_loss": consistency_sum / max(seen, 1),
            },
            "dev": stripped_eval(dev),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (
            float(dev["predicted_router"]["top1_accuracy_↑"]),
            float(dev["oracle_group"]["top1_accuracy_↑"]),
            float(dev["predicted_router"]["top5_accuracy_↑"]),
            float(dev["router_group_accuracy_↑"]),
        )
        if best_key is None or key > best_key:
            best_key = key; best_epoch = epoch; best_dev = stripped_eval(dev)
            best_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_backbone_state = {
                name: value.detach().cpu().clone()
                for name, value in backbone.model.state_dict().items()
                if name in trainable_backbone_names
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break
    if best_model_state is None or best_backbone_state is None or best_dev is None or gradient_audit is None:
        raise RuntimeError("training selected no checkpoint")
    model.load_state_dict(best_model_state)
    state = backbone.model.state_dict()
    state.update(best_backbone_state)
    backbone.model.load_state_dict(state)
    checkpoint_path = output_dir / "target_invariant_atst_v1_best.pt"
    _atomic_torch(
        {
            "format": FORMAT,
            "labels": labels,
            "groups": groups,
            "input_dim": int(previous["input_dim"]),
            "hidden_dim": int(previous["hidden_dim"]),
            "best_epoch": best_epoch,
            "best_dev": best_dev,
            "model_state_dict": best_model_state,
            "atst_trainable_state_dict": best_backbone_state,
            "atst_trainable_parameter_names": trainable_backbone_names,
        },
        checkpoint_path,
    )

    preflight_gates = {
        "source_pool_passed": bool(source_audit["gates"]["passed"]),
        "remix_waveforms_valid": all(
            bool(value) for key, value in remix_audit.items() if key.startswith("all_")
        ),
        "backbone_gradient_nonzero": float(gradient_audit["trainable_gradient_norm_sum"]) > 0,
        "frozen_backbone_has_no_gradient": not gradient_audit["frozen_parameters_with_gradient"],
        "exact_trainable_block_contract": bool(gradient_audit["only_blocks_10_11_and_norm_trainable"]),
    }
    if args.preflight_only or args.skip_matched:
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "preflight; matched evaluation not opened",
            "data": source_audit,
            "remix_audit": remix_audit,
            "gradient_audit": gradient_audit,
            "best_epoch": best_epoch,
            "best_dev": best_dev,
            "history": history,
            "gates": {"passed": all(preflight_gates.values()), "checks": preflight_gates},
            "artifacts": {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "source_pool": str(args.source_pool.resolve()),
                "source_pool_sha256": _sha256_file(args.source_pool.resolve()),
            },
        }
        _atomic_json(receipt, output_dir / "receipt.json")
        print(json.dumps({"complete": True, "preflight_only": True, "gates": receipt["gates"]}, sort_keys=True), flush=True)
        return

    # Model and checkpoint are frozen before matched is read/evaluated.
    matched_rows = load_scene_manifest(args.matched_manifest.resolve(), label_to_id)
    matched = evaluate_rows(
        backbone,
        model,
        matched_rows,
        groups,
        label_to_group,
        device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )
    clean_matched = evaluate_rows(
        backbone,
        model,
        component_rows(matched_rows),
        groups,
        label_to_group,
        device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        component_canvas=True,
    )
    old = old_expert_baseline(previous_path, args.previous_matched_cache.resolve(), device)
    if not torch.equal(old["targets"], matched["targets"]):
        raise RuntimeError("old/new matched event order differs")
    heavy = matched["heavy_mask"]
    old_heavy = ordering_metrics(old["ordering"][heavy], old["targets"][heavy])
    routed_top1 = float(matched["predicted_router"]["top1_accuracy_↑"])
    oracle_top1 = float(matched["oracle_group"]["top1_accuracy_↑"])
    oracle_top5 = float(matched["oracle_group"]["top5_accuracy_↑"])
    clean_top1 = float(clean_matched["oracle_group"]["top1_accuracy_↑"])
    heavy_delta = float(
        matched["heavy_overlap_oracle_group"]["top1_accuracy_↑"] - old_heavy["top1_accuracy_↑"]
    )
    decision_checks = {
        **preflight_gates,
        "clean_matched_oracle_group_top1_ge_0_75": clean_top1 >= 0.75,
        "mixture_matched_oracle_group_top1_ge_0_68": oracle_top1 >= 0.68,
        "mixture_matched_oracle_group_top5_ge_0_93": oracle_top5 >= 0.93,
        "heavy_overlap_oracle_group_gain_ge_0_08": heavy_delta >= 0.08,
        "deployable_router_beats_flat_atst_by_0_05": routed_top1 >= 0.51625 + 0.05,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "ontology_classes": 188,
            "experts": 7,
            "views_per_target": 3,
            "dynamic_remix": "clean + same-confusion-group negative + same-group/random two-negative view",
            "sir_db": [args.sir_min_db, args.sir_max_db],
            "overlap_choices": list(OVERLAP_CHOICES),
            "trainable_backbone": "ATST blocks 10, 11, and norm_frame only",
            "loss_weights": {
                "fine_ce": 1.0,
                "group_ce": args.group_loss_weight,
                "target_identity_nt_xent": args.contrastive_loss_weight,
                "clean_remix_consistency": args.consistency_loss_weight,
            },
        },
        "data": source_audit,
        "remix_audit": remix_audit,
        "gradient_audit": gradient_audit,
        "selection": {
            "matched_used_for_selection": False,
            "primary_key": "dev predicted-router top1, then oracle-group top1/top5 and router group accuracy",
            "best_epoch": best_epoch,
            "best_dev": best_dev,
        },
        "matched_held_out": {
            "new_mixture": stripped_eval(matched),
            "new_clean_components": stripped_eval(clean_matched),
            "previous_oracle_group_experts": old["metrics"],
            "previous_oracle_group_experts_heavy_overlap_ge_0_30": old_heavy,
            "heavy_overlap_oracle_group_top1_delta": heavy_delta,
        },
        "gates": {"passed": all(decision_checks.values()), "checks": decision_checks},
        "decision": "accept_target_invariant_representation" if all(decision_checks.values()) else "stop_188_fine_grained_and_revise_ontology_or_hierarchy",
        "history": history,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "source_pool": str(args.source_pool.resolve()),
            "source_pool_sha256": _sha256_file(args.source_pool.resolve()),
            "ontology_sha256": _sha256_file(args.ontology.resolve()),
            "groups_sha256": sha256_text([str(value) for group in groups for value in group]),
            "previous_experts": str(previous_path),
            "previous_experts_sha256": _sha256_file(previous_path),
            "dev_manifest_sha256": _sha256_file(args.dev_manifest.resolve()),
            "matched_manifest_sha256": _sha256_file(args.matched_manifest.resolve()),
        },
    }
    receipt_path = output_dir / "receipt.json"
    _atomic_json(receipt, receipt_path)
    print(
        json.dumps(
            {
                "complete": True,
                "best_epoch": best_epoch,
                "matched": receipt["matched_held_out"],
                "gates": receipt["gates"],
                "decision": receipt["decision"],
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
