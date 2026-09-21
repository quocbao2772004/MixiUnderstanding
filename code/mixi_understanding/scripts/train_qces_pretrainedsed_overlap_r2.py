#!/usr/bin/env python3
"""R2 detector adaptation with controlled multi-event exposure.

R2 starts from the best R1 event checkpoint.  Every training epoch samples
25% Gold single-event scenes, 25% rendered non-overlapping multi-event scenes,
and 50% on-the-fly hard/overlapping mixtures.  Only BEATs blocks 10/11 and the
strong head are updated.  The fixed-grid target/loss remains unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline" / "PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(value))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from mixi_understanding.qces.clean_evidence_scenes import (
    CleanSource,
    load_source_bank,
    partition_sources,
)
from mixi_understanding.qces.fixed_grid_detector import (
    FIXED_AUDIO_SECONDS,
    FRAME_HOP_SECONDS,
    NUM_FRAMES,
    build_fixed_grid_targets,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    SceneDataset,
    SceneItem,
    interpolate_sequence,
    load_model,
    load_ontology,
    load_scene_manifest,
    make_device,
    set_seed,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector_fixedgrid_r0 import (
    atomic_json,
    atomic_torch,
    compact_summary,
    fixed_grid_pos_weight,
    summarize_predictions,
    threshold_key,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_headtransfer_r1 import (
    clip_pos_weight,
    evaluate,
    row_quality_weight,
    scene_sampling_weights,
    weighted_clip_loss,
    weighted_frame_loss,
)


FORMAT = "qces_pretrainedsed_overlap_r2_v1"
DEFAULT_DATA = PROJECT_ROOT / "outputs" / "qces_full191_r1_data_v1"
DEFAULT_R1 = (
    PROJECT_ROOT
    / "outputs"
    / "qces_pretrainedsed_detector_full191"
    / "headtransfer_r1_v1"
    / "pretrainedsed_beats_qces_detector.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "qces_pretrainedsed_detector_full191" / "overlap_r2_v1"
)
SAMPLE_RATE = 16_000
FIXED_SAMPLES = int(FIXED_AUDIO_SECONDS * SAMPLE_RATE)


HARD_GROUPS = (
    (
        "Artillery_fire", "Cap_gun", "Firecracker", "Machine_gun", "Slam",
        "Slap_and_smack", "Thunk", "Whack_and_thwack", "Clang", "Glass_shatter",
        "Chop", "Basketball_bounce",
    ),
    (
        "Alarm_clock", "Beep_and_bleep", "Busy_signal", "Car_alarm", "Fire_alarm",
        "Reversing_beeps", "Ringtone", "Telephone_bell_ringing", "Dial_tone",
        "Telephone_dialing_and_DTMF",
    ),
    (
        "Air_horn_and_truck_horn", "Civil_defense_siren",
        "Fire_engine_and_fire_truck_(siren)", "Police_car_(siren)", "Train_horn",
        "Train_whistle", "Steam_whistle", "Ice_cream_truck_and_ice_cream_van",
    ),
    (
        "Single-lens_reflex_camera", "Keys_jangling", "Coin_(dropping)",
        "Computer_keyboard", "Typewriter", "Tick", "Tick-tock", "Scissors",
        "Cash_register", "Ratchet_and_pawl", "Clickety-clack",
    ),
    (
        "Accelerating_and_revving_and_vroom", "Bus", "Car_passing_by",
        "Engine_starting", "Heavy_engine_(low_frequency)", "Medium_engine_(mid_frequency)",
        "Idling", "Jet_engine", "Helicopter", "Lawn_mower", "Motorcycle",
        "Motorboat_and_speedboat", "Mechanical_fan", "Vacuum_cleaner",
    ),
    ("Meow", "Purr", "Caterwaul", "Bark", "Growling", "Howl", "Whimper_(dog)"),
    (
        "Chirp_and_tweet", "Caw", "Coo", "Crowing_and_cock-a-doodle-doo",
        "Cluck", "Quack", "Bird_flight_and_flapping_wings",
    ),
    (
        "Drip", "Fill_(with_liquid)", "Gush", "Sink_(filling_or_washing)",
        "Bathtub_(filling_or_washing)", "Stream_and_river", "Trickle_and_dribble",
        "Water_tap_and_faucet", "Waterfall", "Waves_and_surf", "Rain_on_surface",
    ),
    (
        "Female_speech_and_woman_speaking", "Male_speech_and_man_speaking",
        "Child_speech_and_kid_speaking", "Babbling",
        "Hubbub_and_speech_noise_and_speech_babble", "Whispering", "Speech_synthesizer",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--r1-checkpoint", type=Path, default=DEFAULT_R1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="BEATs_strong_1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epoch-samples", type=int, default=8_000)
    parser.add_argument("--overlap-scenes", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--silver-weight", type=float, default=0.35)
    parser.add_argument("--clip-loss-weight", type=float, default=0.20)
    parser.add_argument("--clip-temperature", type=float, default=0.50)
    parser.add_argument("--max-pos-weight", type=float, default=80.0)
    parser.add_argument("--max-clip-pos-weight", type=float, default=20.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    )
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--min-duration", type=float, default=0.08)
    parser.add_argument("--merge-gap", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--max-single-train", type=int, default=0)
    parser.add_argument("--max-multi-train", type=int, default=0)
    parser.add_argument("--max-dev", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "gradient_accumulation", "epoch_samples", "overlap_scenes"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_','-')} must be positive")
    if args.early_stopping_patience < 1 or args.early_stopping_min_delta < 0:
        parser.error("invalid early stopping configuration")
    return args


def stable_seed(seed: int, *values: Any) -> int:
    text = "\0".join([str(seed), *(str(value) for value in values)])
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


class WeightedManifestDataset(Dataset[tuple[torch.Tensor, SceneItem, float]]):
    def __init__(
        self,
        rows: Sequence[SceneItem],
        *,
        silver_weight: float,
        audio_root: Path = Path("/"),
    ) -> None:
        self.rows = list(rows)
        self.base = SceneDataset(self.rows, audio_root=audio_root, target_sample_rate=SAMPLE_RATE)
        self.silver_weight = float(silver_weight)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, SceneItem, float]:
        waveform, row = self.base[index]
        return waveform, row, row_quality_weight(row, self.silver_weight)


class HardOverlapDataset(Dataset[tuple[torch.Tensor, SceneItem, float]]):
    def __init__(
        self,
        sources: Sequence[CleanSource],
        labels: Sequence[str],
        *,
        label_to_id: Mapping[str, int],
        length: int,
        seed: int,
        silver_weight: float,
    ) -> None:
        by_label: dict[str, list[CleanSource]] = defaultdict(list)
        for source in sources:
            if source.label in label_to_id:
                by_label[source.label].append(source)
        missing = [label for label in labels if not by_label[label]]
        if missing:
            raise ValueError(f"overlap source bank misses labels: {missing[:10]}")
        self.by_label = {label: tuple(rows) for label, rows in by_label.items()}
        self.labels = tuple(labels)
        self.label_to_id = dict(label_to_id)
        self.length = int(length)
        self.seed = int(seed)
        self.silver_weight = float(silver_weight)
        self.epoch = 0
        self.groups = tuple(
            tuple(label for label in group if label in self.label_to_id)
            for group in HARD_GROUPS
        )
        self.groups = tuple(group for group in self.groups if len(group) >= 2)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def _labels_for(self, index: int, rng: random.Random) -> list[str]:
        anchor = self.labels[index % len(self.labels)]
        count = rng.randint(3, 6)
        selected = [anchor]
        matching = [group for group in self.groups if anchor in group]
        if matching:
            candidates = [label for label in rng.choice(matching) if label != anchor]
            rng.shuffle(candidates)
            selected.extend(candidates[: min(2, count - 1)])
        elif self.groups and index % 2 == 0:
            pair_group = rng.choice(self.groups)
            selected.extend(rng.sample(list(pair_group), 2))
        remaining = [label for label in self.labels if label not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
        return selected

    def _source_for(self, label: str, rng: random.Random) -> CleanSource:
        return rng.choice(self.by_label[label])

    @staticmethod
    def _plan_starts(lengths: Sequence[int], rng: random.Random) -> list[int]:
        """Place every event fully inside the fixed scene, favouring overlap.

        When the reference event was close to the right boundary, the previous
        implementation could make ``low`` larger than the current event's last
        valid start frame.  ``randint(low, max(low, high))`` then returned an
        out-of-range start and the waveform slice was shorter than the crop.
        """
        starts: list[int] = []
        for event_index, length in enumerate(lengths):
            max_start = NUM_FRAMES - int(length)
            if max_start < 0:
                raise ValueError(f"event length {length} exceeds {NUM_FRAMES} frames")
            if event_index == 0:
                start = rng.randint(0, max_start)
            elif event_index == 1 or rng.random() < 0.65:
                reference = rng.randrange(len(starts))
                ref_start, ref_length = starts[reference], int(lengths[reference])
                low = min(max_start, max(0, ref_start - int(length) // 2))
                high = min(max_start, ref_start + max(1, ref_length // 2))
                high = max(low, high)
                start = rng.randint(low, high)
            else:
                start = rng.randint(0, max_start)
            starts.append(start)
        return starts

    @staticmethod
    def _load_crop(source: CleanSource, rng: random.Random) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(source.audio_path)
        mono = waveform.float().mean(dim=0)
        if int(sample_rate) != SAMPLE_RATE:
            mono = AF.resample(mono, int(sample_rate), SAMPLE_RATE)
        start = max(0, int(round(source.active_onset_seconds * SAMPLE_RATE)))
        end = min(mono.numel(), int(round(source.active_offset_seconds * SAMPLE_RATE)))
        if end <= start:
            raise ValueError(f"invalid active crop for {source.source_id}")
        maximum = int(round(1.50 * SAMPLE_RATE))
        if end - start > maximum:
            start += rng.randint(0, end - start - maximum)
            end = start + maximum
        crop = mono[start:end]
        minimum = int(round(0.08 * SAMPLE_RATE))
        if crop.numel() < minimum:
            crop = F.interpolate(
                crop.view(1, 1, -1), size=minimum, mode="linear", align_corners=False
            ).reshape(-1)
        rms = crop.square().mean().sqrt().clamp_min(1e-5)
        gain_db = rng.choice((-9.0, -6.0, -3.0, 0.0, 3.0))
        return crop / rms * (0.06 * (10.0 ** (gain_db / 20.0)))

    def scene_item(self, index: int) -> SceneItem:
        rng = random.Random(stable_seed(self.seed, self.epoch, index, "metadata"))
        labels = self._labels_for(index, rng)
        sources = [self._source_for(label, rng) for label in labels]
        lengths = [
            max(2, min(38, int(math.ceil(min(source.duration_seconds, 1.5) / FRAME_HOP_SECONDS))))
            for source in sources
        ]
        starts = self._plan_starts(lengths, rng)
        events = []
        for event_index, (label, source, start, length) in enumerate(
            zip(labels, sources, starts, lengths, strict=True)
        ):
            events.append(
                {
                    "event_id": f"r2_overlap_{self.epoch}_{index}:e{event_index}",
                    "label": label,
                    "label_id": self.label_to_id[label],
                    "onset_seconds": start * FRAME_HOP_SECONDS,
                    "offset_seconds": (start + length) * FRAME_HOP_SECONDS,
                    "source_id": source.source_id,
                    "cleanliness_tier": source.cleanliness_tier,
                }
            )
        return SceneItem(
            scene_id=f"r2_overlap_{self.epoch}_{index:07d}",
            split="train",
            mixture_path="on_the_fly",
            duration_seconds=FIXED_AUDIO_SECONDS,
            sample_rate=SAMPLE_RATE,
            events=tuple(events),
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, SceneItem, float]:
        metadata_rng = random.Random(stable_seed(self.seed, self.epoch, index, "metadata"))
        audio_rng = random.Random(stable_seed(self.seed, self.epoch, index, "audio"))
        labels = self._labels_for(index, metadata_rng)
        sources = [self._source_for(label, metadata_rng) for label in labels]
        crops = [self._load_crop(source, audio_rng) for source in sources]
        lengths = [max(2, min(38, int(math.ceil(crop.numel() / SAMPLE_RATE / FRAME_HOP_SECONDS)))) for crop in crops]
        # Fit every crop to the same grid lengths used by the metadata planner.
        metadata_lengths = [
            max(2, min(38, int(math.ceil(min(source.duration_seconds, 1.5) / FRAME_HOP_SECONDS))))
            for source in sources
        ]
        lengths = metadata_lengths
        starts = self._plan_starts(lengths, metadata_rng)
        mixture = torch.zeros(FIXED_SAMPLES, dtype=torch.float32)
        events = []
        hop_samples = int(round(FRAME_HOP_SECONDS * SAMPLE_RATE))
        for event_index, (label, source, crop, start, length) in enumerate(
            zip(labels, sources, crops, starts, lengths, strict=True)
        ):
            target_samples = length * hop_samples
            if crop.numel() != target_samples:
                crop = F.interpolate(
                    crop.view(1, 1, -1), size=target_samples, mode="linear", align_corners=False
                ).reshape(-1)
            start_sample = start * hop_samples
            if start < 0 or start + length > NUM_FRAMES:
                raise RuntimeError(
                    f"invalid placement: index={index}, event={event_index}, "
                    f"start={start}, length={length}, frames={NUM_FRAMES}"
                )
            mixture[start_sample : start_sample + target_samples] += crop
            events.append(
                {
                    "event_id": f"r2_overlap_{self.epoch}_{index}:e{event_index}",
                    "label": label,
                    "label_id": self.label_to_id[label],
                    "onset_seconds": start * FRAME_HOP_SECONDS,
                    "offset_seconds": (start + length) * FRAME_HOP_SECONDS,
                    "source_id": source.source_id,
                    "cleanliness_tier": source.cleanliness_tier,
                }
            )
        peak = mixture.abs().max()
        if peak > 0.95:
            mixture *= 0.95 / peak
        row = SceneItem(
            scene_id=f"r2_overlap_{self.epoch}_{index:07d}",
            split="train",
            mixture_path="on_the_fly",
            duration_seconds=FIXED_AUDIO_SECONDS,
            sample_rate=SAMPLE_RATE,
            events=tuple(events),
        )
        return mixture, row, row_quality_weight(row, self.silver_weight)


def train_collate(
    batch: Sequence[tuple[torch.Tensor, SceneItem, float]],
) -> tuple[torch.Tensor, list[SceneItem], torch.Tensor]:
    return (
        torch.stack([item[0] for item in batch]),
        [item[1] for item in batch],
        torch.tensor([item[2] for item in batch], dtype=torch.float32),
    )


def eval_collate(
    batch: Sequence[tuple[torch.Tensor, SceneItem, float]],
) -> tuple[torch.Tensor, list[SceneItem]]:
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def category_weights(
    single_rows: Sequence[SceneItem],
    multi_rows: Sequence[SceneItem],
    overlap_rows: Sequence[SceneItem],
    labels: Sequence[str],
) -> torch.Tensor:
    # Fixed scene exposure: 25% Gold single, 25% rendered multi, 50% hard overlap.
    pieces = []
    for rows, share in ((single_rows, 0.25), (multi_rows, 0.25), (overlap_rows, 0.50)):
        base = scene_sampling_weights(rows, labels)
        base = base / base.sum().clamp_min(1e-12) * share
        pieces.append(base)
    return torch.cat(pieces)


def set_r2_train_mode(model: torch.nn.Module) -> None:
    model.eval()
    model.strong_head.train()
    found = set()
    for name, module in model.model.named_modules():
        if name in {"beats.encoder.layers.10", "beats.encoder.layers.11"}:
            module.train()
            found.add(name)
    if len(found) != 2:
        raise RuntimeError(f"could not enable exactly two BEATs blocks: {found}")


def forward_logits(model: torch.nn.Module, waveforms: torch.Tensor) -> torch.Tensor:
    mel = model.mel_forward(waveforms)
    features = model.model(mel)
    features = interpolate_sequence(features, model.seq_len)
    features = model.seq_model(features)
    return model.strong_head(features)


@torch.inference_mode()
def collect_audio_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
) -> list[dict[str, Any]]:
    model.eval()
    output = []
    for waveforms, rows in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = forward_logits(model, waveforms)
        logits = logits.detach().cpu().to(torch.float16)
        for index, row in enumerate(rows):
            output.append(
                {
                    "scene_id": row.scene_id,
                    "duration_seconds": row.duration_seconds,
                    "valid_frames": NUM_FRAMES,
                    "gold_events": list(row.events),
                    "logits": logits[index],
                }
            )
    return output


def evaluate_audio(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    labels: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    predictions = collect_audio_predictions(model, loader, device=device, amp=args.amp)
    summaries = [
        summarize_predictions(
            predictions,
            labels,
            threshold=threshold,
            event_iou_threshold=args.event_iou_threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
        )
        for threshold in args.thresholds
    ]
    selected = {
        objective: max(summaries, key=lambda row, name=objective: threshold_key(row, name))
        for objective in ("event", "frame", "scene")
    }
    return predictions, summaries, selected


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    set_seed(args.seed)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ontology_path = next(data_dir.glob("ontology_*.txt"))
    labels = load_ontology(ontology_path)
    label_to_id = {label: index for index, label in enumerate(labels)}
    single_rows_all = load_scene_manifest(
        data_dir / "detector_scene_manifest_single_train.jsonl",
        label_to_id,
        args.max_single_train,
    )
    single_rows = [
        row for row in single_rows_all
        if len(row.events) == 1
        and str(row.events[0].get("cleanliness_tier") or "").lower() == "gold"
    ]
    multi_rows = load_scene_manifest(
        data_dir / "detector_scene_manifest_multi_train.jsonl",
        label_to_id,
        args.max_multi_train,
    )
    dev_rows = load_scene_manifest(
        data_dir / "detector_scene_manifest_multi_dev.jsonl", label_to_id, args.max_dev
    )
    test_rows = load_scene_manifest(
        data_dir / "detector_scene_manifest_multi_test.jsonl", label_to_id, args.max_test
    )
    source_bank = load_source_bank(data_dir / "source_bank_accepted.jsonl", require_audio_file=True)
    partitioned, _ = partition_sources(source_bank, labels, seed=2041, dev_fraction=0.20)
    overlap_dataset = HardOverlapDataset(
        partitioned["train"],
        labels,
        label_to_id=label_to_id,
        length=args.overlap_scenes,
        seed=args.seed,
        silver_weight=args.silver_weight,
    )
    overlap_rows = [overlap_dataset.scene_item(index) for index in range(len(overlap_dataset))]

    single_dataset = WeightedManifestDataset(
        single_rows, silver_weight=args.silver_weight, audio_root=Path("/")
    )
    multi_dataset = WeightedManifestDataset(
        multi_rows, silver_weight=args.silver_weight, audio_root=Path("/")
    )
    train_dataset = ConcatDataset([single_dataset, multi_dataset, overlap_dataset])
    sampler_weights = category_weights(single_rows, multi_rows, overlap_rows, labels)
    sampler = WeightedRandomSampler(
        sampler_weights,
        num_samples=args.epoch_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_collate,
    )
    dev_loader = DataLoader(
        WeightedManifestDataset(dev_rows, silver_weight=args.silver_weight),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=eval_collate,
    )
    test_loader = DataLoader(
        WeightedManifestDataset(test_rows, silver_weight=args.silver_weight),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=eval_collate,
    )

    device = make_device(args.device)
    model = load_model(len(labels), args.checkpoint, device, unfreeze_last_blocks=0)
    r1 = torch.load(args.r1_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if r1.get("labels") != labels:
        raise ValueError("R1 ontology mismatch")
    model.load_state_dict(r1["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.strong_head.requires_grad_(True)
    backbone_parameters = []
    for name, parameter in model.model.named_parameters():
        if "beats.encoder.layers.10." in name or "beats.encoder.layers.11." in name:
            parameter.requires_grad = True
            backbone_parameters.append(parameter)
    if not backbone_parameters:
        raise RuntimeError("last two BEATs blocks were not found")
    unexpected = [name for name, parameter in model.model.named_parameters() if parameter.requires_grad and not (
        "beats.encoder.layers.10." in name or "beats.encoder.layers.11." in name
    )]
    if unexpected:
        raise RuntimeError(f"unexpected trainable backbone parameters: {unexpected[:5]}")

    # Fixed class imbalance statistics from the explicit 25/25/50 curriculum pool.
    supervision_rows = single_rows + multi_rows + overlap_rows
    frame_pos_weight = fixed_grid_pos_weight(
        supervision_rows, len(labels), max_pos_weight=args.max_pos_weight
    ).to(device)
    scene_pos_weight = clip_pos_weight(
        supervision_rows, labels, args.max_clip_pos_weight
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.strong_head.parameters(), "lr": args.head_lr},
            {"params": backbone_parameters, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = (-math.inf,)
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    best_path = output_dir / "best_event_running.pt"
    history = []
    stale_epochs = 0
    monitor_best = -math.inf

    def evaluate_and_save(epoch: int, train_loss: float | None) -> float:
        nonlocal best_score, best_epoch, best_metrics
        _, _, selected = evaluate_audio(
            model, dev_loader, labels=labels, args=args, device=device
        )
        event = selected["event"]
        score = threshold_key(event, "event")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = compact_summary(event)
            atomic_torch(
                best_path,
                {
                    "format": FORMAT,
                    "epoch": epoch,
                    "labels": labels,
                    "metrics": best_metrics,
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                },
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selected": {name: compact_summary(value) for name, value in selected.items()},
            "best_epoch": best_epoch,
        }
        history.append(row)
        atomic_json(
            output_dir / "progress.json",
            {"format": FORMAT, "status": "running", "history": history, "best": best_metrics},
        )
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        return float(event["event_iou"]["f1_↑"])

    print(
        json.dumps(
            {
                "labels": len(labels),
                "single_gold": len(single_rows),
                "rendered_multi": len(multi_rows),
                "on_the_fly_overlap": len(overlap_dataset),
                "epoch_samples": args.epoch_samples,
                "exposure": {"single_gold": 0.25, "rendered_multi": 0.25, "hard_overlap": 0.50},
                "trainable_backbone_blocks": [10, 11],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    baseline_f1 = evaluate_and_save(0, None)
    monitor_best = baseline_f1

    trainable = list(model.strong_head.parameters()) + backbone_parameters
    for epoch in range(1, args.epochs + 1):
        overlap_dataset.set_epoch(epoch)
        set_r2_train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (waveforms, rows, quality_weights) in enumerate(train_loader, start=1):
            waveforms = waveforms.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = forward_logits(model, waveforms)
                targets, valid = build_fixed_grid_targets(
                    rows, num_labels=len(labels), device=device
                )
                frame_loss = weighted_frame_loss(
                    logits, targets, valid, quality_weights, frame_pos_weight
                )
                presence_loss = weighted_clip_loss(
                    logits,
                    targets,
                    valid,
                    quality_weights,
                    scene_pos_weight,
                    args.clip_temperature,
                )
                loss = frame_loss + args.clip_loss_weight * presence_loss
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if step % 250 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} loss={np.mean(losses[-250:]):.5f}",
                    flush=True,
                )
        event_f1 = evaluate_and_save(epoch, float(np.mean(losses)))
        if event_f1 > monitor_best + args.early_stopping_min_delta:
            monitor_best = event_f1
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.early_stopping_patience:
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "patience": args.early_stopping_patience,
                        "monitor_best_event_f1": monitor_best,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    _, dev_summaries, dev_selected = evaluate_audio(
        model, dev_loader, labels=labels, args=args, device=device
    )
    threshold = float(dev_selected["event"]["threshold"])
    test_predictions = collect_audio_predictions(model, test_loader, device=device, amp=args.amp)
    test_summary = summarize_predictions(
        test_predictions,
        labels,
        threshold=threshold,
        event_iou_threshold=args.event_iou_threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
    )
    report = {
        "format": FORMAT,
        "status": "complete",
        "initialization": str(args.r1_checkpoint.resolve()),
        "labels": len(labels),
        "curriculum": {
            "epoch_samples": args.epoch_samples,
            "single_gold_share": 0.25,
            "rendered_multi_share": 0.25,
            "hard_overlap_share": 0.50,
            "hard_overlap_dynamic_each_epoch": True,
        },
        "optimization": {
            "maximum_epochs": args.epochs,
            "completed_epochs": len(history) - 1,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "trainable_backbone_blocks": [10, 11],
        },
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_dev": dev_selected["event"],
        "dev_threshold_summaries": dev_summaries,
        "locked_test": test_summary,
    }
    atomic_json(output_dir / "training_report.json", report)
    atomic_torch(
        output_dir / "pretrainedsed_beats_qces_detector_r2.pt",
        {
            "format": FORMAT,
            "labels": labels,
            "model_state_dict": model.state_dict(),
            "best_epoch": int(best["epoch"]),
            "best_threshold": threshold,
            "report": report,
        },
    )
    atomic_json(
        output_dir / "progress.json",
        {
            "format": FORMAT,
            "status": "complete",
            "best_epoch": int(best["epoch"]),
            "best_dev": compact_summary(dev_selected["event"]),
            "locked_test": compact_summary(test_summary),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "best_epoch": int(best["epoch"]),
                "best_dev": compact_summary(dev_selected["event"]),
                "locked_test": compact_summary(test_summary),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
