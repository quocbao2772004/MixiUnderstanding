#!/usr/bin/env python3
"""Trust-region BEATs adaptation for event-label retrieval in overlap scenes.

This experiment is intentionally different from the historical overlap-R2
run.  R2 optimized dense frame/clip BCE, updated the strong head, increased
scene recall but collapsed precision, and selected epoch zero.  Here:

* the proven 191-way strong head is frozen;
* only the final BEATs blocks are updated at a small learning rate;
* supervision ranks each gold event above absent hard-negative labels in its
  own temporal interval, without making simultaneous gold labels compete;
* cached R1 logits impose anti-false-positive and non-overlap retention terms.

The output is a sidecar checkpoint and never overwrites R1 or RED/EPN.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline" / "PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from mixi_understanding.qces.benchmark_integrity import semantic_events
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    interpolate_sequence,
    load_model,
    make_device,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector_fixedgrid_r0 import (
    atomic_json,
    atomic_torch,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    NUM_FRAMES,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import (
    assert_dense_identity_disjoint_v2,
)


FORMAT = "qces_beats_event_rank_v1"
SAMPLE_RATE = 16_000
FIXED_SAMPLES = 10 * SAMPLE_RATE


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    multi = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/multievent"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--nonoverlap-train-index", type=Path, default=base / "dense_multi_train_v2/index.json")
    parser.add_argument("--nonoverlap-dev-index", type=Path, default=base / "dense_multi_dev_v2/index.json")
    parser.add_argument("--overlap-train-index", type=Path, default=base / "dense_overlap_query_train_v2/index.json")
    parser.add_argument("--overlap-dev-index", type=Path, default=base / "dense_overlap_query_dev_v2/index.json")
    parser.add_argument("--nonoverlap-train-scenes", type=Path, default=multi / "scene_ids_train.txt")
    parser.add_argument("--nonoverlap-dev-scenes", type=Path, default=multi / "scene_ids_dev.txt")
    parser.add_argument("--overlap-train-scenes", type=Path, default=overlap / "scene_ids_overlap_train.txt")
    parser.add_argument("--overlap-dev-scenes", type=Path, default=overlap / "scene_ids_overlap_dev.txt")
    parser.add_argument("--output-dir", type=Path, default=base / "beats_event_rank_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained-checkpoint", default="BEATs_strong_1")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--hard-negatives", type=int, default=12)
    parser.add_argument("--scene-rank-weight", type=float, default=0.35)
    parser.add_argument("--anti-fp-weight", type=float, default=0.05)
    parser.add_argument("--retention-weight", type=float, default=0.02)
    parser.add_argument("--l2sp-weight", type=float, default=1e-3)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2173)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class RankScene:
    scene_id: str
    domain: str
    mixture_path: Path
    valid_frames: int
    events: tuple[dict[str, Any], ...]


class AudioTeacherDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        store: DenseFeatureStore,
        scene_ids: Sequence[str],
        *,
        domain: str,
        label_to_id: Mapping[str, int],
        max_scenes: int = 0,
    ) -> None:
        selected = list(dict.fromkeys(str(value) for value in scene_ids))
        if max_scenes > 0:
            selected = selected[:max_scenes]
        if not selected:
            raise ValueError("event-rank dataset cannot be empty")
        self.rows: list[RankScene] = []
        self.teacher_logits: list[torch.Tensor] = []
        for scene_id in selected:
            metadata = store.metadata(scene_id)
            path = Path(str(metadata["mixture_path"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            events = []
            for event in semantic_events(metadata):
                label = str(event["label"])
                if label not in label_to_id:
                    raise ValueError(f"{scene_id}: unknown event label {label}")
                events.append(
                    dict(event)
                    | {
                        "label_id": int(label_to_id[label]),
                        "onset_seconds": float(event["onset_seconds"]),
                        "offset_seconds": float(event["offset_seconds"]),
                    }
                )
            if not events:
                raise ValueError(f"{scene_id}: no semantic events")
            self.rows.append(
                RankScene(
                    scene_id=scene_id,
                    domain=domain,
                    mixture_path=path,
                    valid_frames=int(metadata["valid_frames"]),
                    events=tuple(events),
                )
            )
            # Clone only logits. Keeping a shard view here would retain the much
            # larger frozen feature tensor for every shard.
            self.teacher_logits.append(store.get(scene_id)["logits"].clone())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, sample_rate = torchaudio.load(row.mixture_path)
        waveform = waveform.float().mean(dim=0)
        if int(sample_rate) != SAMPLE_RATE:
            waveform = AF.resample(waveform, int(sample_rate), SAMPLE_RATE)
        if waveform.numel() < FIXED_SAMPLES:
            waveform = F.pad(waveform, (0, FIXED_SAMPLES - waveform.numel()))
        elif waveform.numel() > FIXED_SAMPLES:
            waveform = waveform[:FIXED_SAMPLES]
        return {
            "waveform": waveform,
            "teacher_logits": self.teacher_logits[index].float(),
            "scene": row,
        }


def collate_rank(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "waveform": torch.stack([row["waveform"] for row in rows]),
        "teacher_logits": torch.stack([row["teacher_logits"] for row in rows]),
        "scenes": [row["scene"] for row in rows],
    }


def forward_logits(model: torch.nn.Module, waveforms: torch.Tensor) -> torch.Tensor:
    mel = model.mel_forward(waveforms)
    features = model.model(mel)
    features = interpolate_sequence(features, model.seq_len)
    features = model.seq_model(features)
    return model.strong_head(features)


def frame_targets(
    scenes: Sequence[RankScene], num_classes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(len(scenes), NUM_FRAMES, num_classes, device=device)
    valid = torch.zeros(len(scenes), NUM_FRAMES, dtype=torch.bool, device=device)
    for batch_index, scene in enumerate(scenes):
        valid[batch_index, : scene.valid_frames] = True
        for event in scene.events:
            start = max(0, min(scene.valid_frames - 1, int(math.floor(event["onset_seconds"] / 0.04))))
            end = max(start + 1, min(scene.valid_frames, int(math.ceil(event["offset_seconds"] / 0.04))))
            target[batch_index, start:end, int(event["label_id"])] = 1.0
    return target, valid


def _event_frame_bounds(event: Mapping[str, Any], valid_frames: int) -> tuple[int, int]:
    start = max(0, min(valid_frames - 1, int(math.floor(float(event["onset_seconds"]) / 0.04))))
    end = max(start + 1, min(valid_frames, int(math.ceil(float(event["offset_seconds"]) / 0.04))))
    return start, end


def event_hard_negative_loss(
    logits: torch.Tensor,
    scenes: Sequence[RankScene],
    *,
    margin: float,
    hard_negatives: int,
) -> torch.Tensor:
    losses = []
    num_classes = logits.shape[-1]
    for batch_index, scene in enumerate(scenes):
        for target_event in scene.events:
            start, end = _event_frame_bounds(target_event, scene.valid_frames)
            pooled = logits[batch_index, start:end].mean(dim=0)
            excluded = torch.zeros(num_classes, dtype=torch.bool, device=logits.device)
            target_onset = float(target_event["onset_seconds"])
            target_offset = float(target_event["offset_seconds"])
            for event in scene.events:
                if min(target_offset, float(event["offset_seconds"])) > max(
                    target_onset, float(event["onset_seconds"])
                ):
                    excluded[int(event["label_id"])] = True
            negatives = pooled.masked_fill(excluded, -torch.inf)
            count = min(hard_negatives, int((~excluded).sum().item()))
            if count <= 0:
                continue
            negative_scores = negatives.topk(count).values
            positive_score = pooled[int(target_event["label_id"])]
            losses.append(F.softplus(margin + negative_scores - positive_score).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def scene_hard_negative_loss(
    logits: torch.Tensor,
    scenes: Sequence[RankScene],
    *,
    margin: float,
    hard_negatives: int,
) -> torch.Tensor:
    losses = []
    for batch_index, scene in enumerate(scenes):
        pooled = logits[batch_index, : scene.valid_frames].amax(dim=0)
        positives = sorted({int(event["label_id"]) for event in scene.events})
        excluded = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        excluded[positives] = True
        count = min(hard_negatives, int((~excluded).sum().item()))
        negative_scores = pooled.masked_fill(excluded, -torch.inf).topk(count).values
        for label_id in positives:
            losses.append(F.softplus(margin + negative_scores - pooled[label_id]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def trust_region_losses(
    student: torch.Tensor,
    teacher: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    scenes: Sequence[RankScene],
) -> tuple[torch.Tensor, torch.Tensor]:
    valid3 = valid.unsqueeze(-1)
    inactive = (target < 0.5) & valid3
    # Do not let adaptation increase an absent class above its R1 logit. It may
    # lower an R1 false positive, which is exactly what overlap-R2 lacked.
    overshoot = F.relu(student - teacher)
    anti_fp = overshoot.square()[inactive].mean()

    nonoverlap = torch.tensor(
        [scene.domain == "nonoverlap" for scene in scenes],
        dtype=torch.bool,
        device=student.device,
    )[:, None, None]
    retention_mask = valid3 & nonoverlap
    expanded_retention = retention_mask.expand_as(student)
    if expanded_retention.any():
        retention = F.smooth_l1_loss(
            student[expanded_retention],
            teacher[expanded_retention],
        )
    else:
        # A shuffled mini-batch may contain overlap scenes only. Returning a
        # differentiable zero avoids the empty-tensor mean becoming NaN.
        retention = student.sum() * 0.0
    return anti_fp, retention


@torch.inference_mode()
def evaluate_semantics(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, *, amp: bool
) -> dict[str, Any]:
    model.eval()
    totals: Counter[str] = Counter()
    rank_sum = 0.0
    for batch in loader:
        waveforms = batch["waveform"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = forward_logits(model, waveforms)
        for batch_index, scene in enumerate(batch["scenes"]):
            totals["scenes"] += 1
            scene_order = logits[batch_index, : scene.valid_frames].amax(dim=0).argsort(descending=True)
            all_scene = {k: True for k in (1, 5, 8, 20)}
            for event in scene.events:
                totals["events"] += 1
                label_id = int(event["label_id"])
                start, end = _event_frame_bounds(event, scene.valid_frames)
                ordering = logits[batch_index, start:end].mean(dim=0).argsort(descending=True)
                rank = int(torch.where(ordering == label_id)[0][0].item()) + 1
                rank_sum += rank
                for top_k in (1, 5, 8, 20):
                    interval_hit = rank <= top_k
                    scene_hit = bool((scene_order[:top_k] == label_id).any())
                    totals[f"interval_top{top_k}"] += int(interval_hit)
                    totals[f"scene_top{top_k}"] += int(scene_hit)
                    all_scene[top_k] = all_scene[top_k] and scene_hit
            for top_k, hit in all_scene.items():
                totals[f"all_scene_top{top_k}"] += int(hit)
    events = max(int(totals["events"]), 1)
    scenes = max(int(totals["scenes"]), 1)
    result: dict[str, Any] = {
        "scenes": int(totals["scenes"]),
        "gold_events": int(totals["events"]),
        "mean_oracle_interval_label_rank_↓": rank_sum / events,
    }
    for top_k in (1, 5, 8, 20):
        result[f"oracle_interval_top{top_k}_accuracy_↑"] = totals[f"interval_top{top_k}"] / events
        result[f"scene_label_top{top_k}_recall_↑"] = totals[f"scene_top{top_k}"] / events
        result[f"all_gold_scene_label_top{top_k}_accuracy_↑"] = totals[f"all_scene_top{top_k}"] / scenes
    return result


def selection_score(
    nonoverlap: Mapping[str, Any], overlap: Mapping[str, Any], baseline_nonoverlap: Mapping[str, Any]
) -> float:
    overlap8 = float(overlap["oracle_interval_top8_accuracy_↑"])
    overlap1 = float(overlap["oracle_interval_top1_accuracy_↑"])
    non8 = float(nonoverlap["oracle_interval_top8_accuracy_↑"])
    baseline_non8 = float(baseline_nonoverlap["oracle_interval_top8_accuracy_↑"])
    retention_penalty = 2.0 * max(0.0, baseline_non8 - 0.01 - non8)
    return overlap8 + 0.25 * overlap1 + 0.25 * non8 - retention_penalty


def main() -> None:
    args = parse_args()
    if not 1 <= args.unfreeze_last_blocks <= 4:
        raise SystemExit("--unfreeze-last-blocks must be in [1,4]")
    for name in ("epochs", "patience", "batch_size", "eval_batch_size", "gradient_accumulation"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    indexes = [
        args.nonoverlap_train_index.resolve(),
        args.nonoverlap_dev_index.resolve(),
        args.overlap_train_index.resolve(),
        args.overlap_dev_index.resolve(),
    ]
    store = DenseFeatureStore(indexes, cache_size=8)
    labels = list(store.labels or [])
    label_to_id = {label: index for index, label in enumerate(labels)}
    ids = {
        "nonoverlap_train": load_scene_list(args.nonoverlap_train_scenes),
        "nonoverlap_dev": load_scene_list(args.nonoverlap_dev_scenes),
        "overlap_train": load_scene_list(args.overlap_train_scenes),
        "overlap_dev": load_scene_list(args.overlap_dev_scenes),
    }
    identity = assert_dense_identity_disjoint_v2(
        store,
        {
            "train": ids["nonoverlap_train"] + ids["overlap_train"],
            "dev": ids["nonoverlap_dev"] + ids["overlap_dev"],
        },
    )
    datasets = {
        name: AudioTeacherDataset(
            store,
            scene_ids,
            domain="overlap" if name.startswith("overlap") else "nonoverlap",
            label_to_id=label_to_id,
            max_scenes=args.max_train_scenes if name.endswith("train") else args.max_dev_scenes,
        )
        for name, scene_ids in ids.items()
    }
    if len(datasets["nonoverlap_train"]) != len(datasets["overlap_train"]):
        raise RuntimeError("training requires equal non-overlap and overlap exposure")
    train_loader = DataLoader(
        ConcatDataset((datasets["nonoverlap_train"], datasets["overlap_train"])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_rank,
    )
    eval_loader_args = {
        "batch_size": args.eval_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate_rank,
    }
    dev_loaders = {
        "nonoverlap": DataLoader(datasets["nonoverlap_dev"], **eval_loader_args),
        "overlap": DataLoader(datasets["overlap_dev"], **eval_loader_args),
    }

    device = make_device(args.device)
    model = load_model(len(labels), args.pretrained_checkpoint, device, unfreeze_last_blocks=0)
    r1_path = args.r1_checkpoint.resolve()
    r1 = torch.load(r1_path, map_location="cpu", weights_only=True)
    if list(r1.get("labels") or []) != labels:
        raise ValueError("R1 ontology mismatch")
    model.load_state_dict(r1["model_state_dict"], strict=True)
    model.requires_grad_(False)
    first_block = 12 - args.unfreeze_last_blocks
    trainable_names = []
    for name, parameter in model.model.named_parameters():
        if any(f"beats.encoder.layers.{index}." in name for index in range(first_block, 12)):
            parameter.requires_grad = True
            trainable_names.append(f"model.{name}")
    if not trainable_names or any(parameter.requires_grad for parameter in model.strong_head.parameters()):
        raise RuntimeError("trainable-parameter contract failed")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(trainable, lr=args.backbone_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    def evaluate_all() -> dict[str, Any]:
        return {
            name: evaluate_semantics(model, loader, device, amp=args.amp)
            for name, loader in dev_loaders.items()
        }

    baseline = evaluate_all()
    best_score = selection_score(baseline["nonoverlap"], baseline["overlap"], baseline["nonoverlap"])
    best_epoch = 0
    best_metrics = baseline
    checkpoint_path = output_dir / "beats_event_rank_best.pt"
    atomic_torch(
        checkpoint_path,
        {
            "format": FORMAT,
            "epoch": 0,
            "labels": labels,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metrics": baseline,
            "base_r1_checkpoint": str(r1_path),
        },
    )
    print(json.dumps({"epoch": 0, "metrics": baseline, "selection_score": best_score}, sort_keys=True), flush=True)

    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.eval()
        for name, module in model.model.named_modules():
            if any(name == f"beats.encoder.layers.{index}" for index in range(first_block, 12)):
                module.train()
        optimizer.zero_grad(set_to_none=True)
        totals: Counter[str] = Counter()
        steps = 0
        for step, batch in enumerate(train_loader, 1):
            waveforms = batch["waveform"].to(device, non_blocking=True)
            teacher = batch["teacher_logits"].to(device, non_blocking=True)
            scenes = batch["scenes"]
            target, valid = frame_targets(scenes, len(labels), device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                student = forward_logits(model, waveforms)
                event_rank = event_hard_negative_loss(
                    student, scenes, margin=args.margin, hard_negatives=args.hard_negatives
                )
                scene_rank = scene_hard_negative_loss(
                    student, scenes, margin=args.margin, hard_negatives=args.hard_negatives
                )
                anti_fp, retention = trust_region_losses(student, teacher, target, valid, scenes)
                l2sp = torch.stack(
                    [
                        (parameter - initial[name]).square().mean()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    ]
                ).mean()
                loss = (
                    event_rank
                    + args.scene_rank_weight * scene_rank
                    + args.anti_fp_weight * anti_fp
                    + args.retention_weight * retention
                    + args.l2sp_weight * l2sp
                )
            if not torch.isfinite(loss):
                diagnostics = {
                    "student_finite": bool(torch.isfinite(student).all()),
                    "student_abs_max": float(student.detach().abs().nan_to_num().max()),
                    "teacher_finite": bool(torch.isfinite(teacher).all()),
                    "event_rank": float(event_rank.detach()),
                    "scene_rank": float(scene_rank.detach()),
                    "anti_fp": float(anti_fp.detach()),
                    "retention": float(retention.detach()),
                    "l2sp": float(l2sp.detach()),
                }
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch} step={step}: {diagnostics}"
                )
            scaler.scale(loss / args.gradient_accumulation).backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for name, value in {
                "loss": loss,
                "event_rank": event_rank,
                "scene_rank": scene_rank,
                "anti_fp": anti_fp,
                "retention": retention,
                "l2sp": l2sp,
            }.items():
                totals[name] += float(value.detach())
            steps += 1
            if step % 250 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={totals['loss']/steps:.5f}", flush=True)
        metrics = evaluate_all()
        score = selection_score(metrics["nonoverlap"], metrics["overlap"], baseline["nonoverlap"])
        row = {
            "epoch": epoch,
            "train": {name: value / max(steps, 1) for name, value in totals.items()},
            "metrics": metrics,
            "selection_score": score,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            stale = 0
            atomic_torch(
                checkpoint_path,
                {
                    "format": FORMAT,
                    "epoch": epoch,
                    "labels": labels,
                    "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "metrics": metrics,
                    "base_r1_checkpoint": str(r1_path),
                },
            )
        else:
            stale += 1
        if stale >= args.patience:
            break

    overlap_base = baseline["overlap"]
    overlap_best = best_metrics["overlap"]
    non_base = baseline["nonoverlap"]
    non_best = best_metrics["nonoverlap"]
    gates = {
        "overlap_interval_top8_gain_ge_0_05": float(overlap_best["oracle_interval_top8_accuracy_↑"])
        >= float(overlap_base["oracle_interval_top8_accuracy_↑"]) + 0.05,
        "overlap_interval_top20_gain_ge_0_03": float(overlap_best["oracle_interval_top20_accuracy_↑"])
        >= float(overlap_base["oracle_interval_top20_accuracy_↑"]) + 0.03,
        "overlap_interval_top1_gain_ge_0_02": float(overlap_best["oracle_interval_top1_accuracy_↑"])
        >= float(overlap_base["oracle_interval_top1_accuracy_↑"]) + 0.02,
        "nonoverlap_interval_top8_drop_le_0_01": float(non_best["oracle_interval_top8_accuracy_↑"])
        >= float(non_base["oracle_interval_top8_accuracy_↑"]) - 0.01,
    }
    report = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "labels": len(labels),
        "base_r1_checkpoint": str(r1_path),
        "base_r1_sha256": _sha256_file(r1_path),
        "answer_label_used_as_input": False,
        "head_frozen": True,
        "trainable_backbone_blocks": list(range(first_block, 12)),
        "trainable_parameter_names": trainable_names,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {name: len(dataset) for name, dataset in datasets.items()},
        "identity_audit": identity,
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_selection_score": best_score,
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
    }
    atomic_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "complete": True,
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "all_success_gates_pass": report["all_success_gates_pass"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
