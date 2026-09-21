#!/usr/bin/env python3
"""Train a question-conditioned acoustic anchor/answer reranker.

This is a sidecar experiment: the detector, overlap-aware proposal generator,
dense caches, and data split remain frozen.  Unlike the feature-only v1
ranker, this model:

* keeps every predicted anchor occurrence instead of fixing the first one;
* scores all anchor/answer pairs without hard-discarding onset-order errors;
* encodes the BEATs frames inside each proposed interval;
* represents no-evidence as ``(localized anchor, NONE)`` rather than a global
  abstention with no verifiable anchor.

Gold answer labels and intervals are targets only.  They are never used to
create proposals or as model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from mixi_understanding.qces.class_aware_red_epn_v1 import (
    ClassAwareRedEpnV1,
    ClassAwareRedEpnV1Config,
    decode_class_aware_proposals,
    interval_iou,
)
from mixi_understanding.scripts.audit_qces_overlap_aware_red_epn_v2 import (
    adjacent_index_pairs,
)
from mixi_understanding.scripts.train_qces_class_aware_red_epn_v1 import (
    ClassAwareSceneDataset,
    collate_class_aware,
)
from mixi_understanding.scripts.train_qces_gold_natural_v3_listwise_ranker import (
    DEFAULT_BASE,
    DEFAULT_DATA,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)


FORMAT = "qces_gold_natural_v3_acoustic_pair_ranker_receipt_v2"
CHECKPOINT_FORMAT = "qces_gold_natural_v3_acoustic_pair_ranker_checkpoint_v2"
NUMERIC_DIM = 20
MAX_EVENTS_PER_SCENE = 8


@dataclass(frozen=True)
class AcousticRankerConfig:
    feature_dim: int
    num_classes: int
    max_proposals: int = 120
    max_pairs: int = 360
    max_anchors: int = 3
    span_samples: int = 12
    span_hidden_dim: int = 128
    span_dim: int = 96
    label_embedding_dim: int = 32
    relation_embedding_dim: int = 4
    numeric_dim: int = NUMERIC_DIM
    rank_hidden_dim: int = 192
    dropout: float = 0.10

    @property
    def list_size(self) -> int:
        return self.max_pairs + self.max_anchors


class AcousticPairNoneRanker(nn.Module):
    """Span encoder plus question-conditioned pair/NONE scoring heads."""

    def __init__(self, config: AcousticRankerConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.span_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.frame_attention = nn.Linear(config.span_hidden_dim, 1)
        self.span_projection = nn.Sequential(
            nn.LayerNorm(3 * config.span_hidden_dim),
            nn.Linear(3 * config.span_hidden_dim, config.span_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.semantic_classifier = nn.Linear(config.span_dim, config.num_classes)
        self.label_embedding = nn.Embedding(
            config.num_classes, config.label_embedding_dim
        )
        self.relation_embedding = nn.Embedding(2, config.relation_embedding_dim)
        semantic_dim = 3
        pair_dim = (
            config.numeric_dim
            + 4 * config.span_dim
            + 2 * config.label_embedding_dim
            + config.relation_embedding_dim
            + 2 * semantic_dim
        )
        none_dim = (
            config.numeric_dim
            + config.span_dim
            + config.label_embedding_dim
            + config.relation_embedding_dim
            + semantic_dim
        )
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, config.rank_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.rank_hidden_dim, config.rank_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.rank_hidden_dim // 2, 1),
        )
        self.none_mlp = nn.Sequential(
            nn.LayerNorm(none_dim),
            nn.Linear(none_dim, config.rank_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.rank_hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)
        nn.init.zeros_(self.none_mlp[-1].weight)
        nn.init.zeros_(self.none_mlp[-1].bias)

    def encode_spans(
        self,
        features: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample frames inside intervals and produce acoustic/semantic reps."""

        if features.ndim != 3 or starts.shape != ends.shape or starts.ndim != 2:
            raise ValueError("bad scene feature/span shapes")
        batch, frames, dimension = features.shape
        if dimension != self.config.feature_dim or starts.shape[0] != batch:
            raise ValueError("scene feature/span dimensions disagree with config")
        positions = torch.linspace(
            0.04,
            0.96,
            self.config.span_samples,
            device=features.device,
            dtype=features.dtype,
        )
        sample_position = starts[..., None] + (
            ends - starts
        )[..., None] * positions
        frame_index = (sample_position * frames).floor().long().clamp(0, frames - 1)
        batch_offset = (
            torch.arange(batch, device=features.device)[:, None, None] * frames
        )
        sampled = features.reshape(batch * frames, dimension)[
            frame_index + batch_offset
        ]
        hidden = self.frame_encoder(sampled)
        mean = hidden.mean(dim=2)
        maximum = hidden.amax(dim=2)
        attention = torch.softmax(
            self.frame_attention(hidden).squeeze(-1), dim=2
        )
        attended = (hidden * attention[..., None]).sum(dim=2)
        representation = self.span_projection(
            torch.cat((mean, maximum, attended), dim=-1)
        )
        representation = representation * mask[..., None].to(representation.dtype)
        semantic = self.semantic_classifier(representation)
        return representation, semantic

    @staticmethod
    def _gather_bank(bank: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        clamped = index.long().clamp_min(0)
        suffix = bank.shape[2:]
        gather_index = clamped.reshape(*clamped.shape, *([1] * len(suffix)))
        gather_index = gather_index.expand(*clamped.shape, *suffix)
        return torch.gather(bank, 1, gather_index)

    @staticmethod
    def _semantic_summary(
        logits: torch.Tensor, label_id: torch.Tensor
    ) -> torch.Tensor:
        own = logits.gather(-1, label_id.long()[..., None]).squeeze(-1)
        top = logits.topk(k=2, dim=-1)
        other = torch.where(
            top.indices[..., 0] == label_id.long(),
            top.values[..., 1],
            top.values[..., 0],
        )
        return torch.stack((own, own.sigmoid(), own - other), dim=-1)

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        proposal_rep, proposal_semantic = self.encode_spans(
            batch["scene_features"],
            batch["proposal_start"],
            batch["proposal_end"],
            batch["proposal_mask"],
        )
        anchor_index = batch["pair_anchor_index"]
        answer_index = batch["pair_answer_index"]
        anchor_rep = self._gather_bank(proposal_rep, anchor_index)
        answer_rep = self._gather_bank(proposal_rep, answer_index)
        anchor_semantic = self._gather_bank(proposal_semantic, anchor_index)
        answer_semantic = self._gather_bank(proposal_semantic, answer_index)
        anchor_label_id = batch["anchor_label_id"][:, None].expand_as(anchor_index)
        answer_label_id = self._gather_bank(
            batch["proposal_label_id"][..., None], answer_index
        ).squeeze(-1)
        relation_id = batch["relation_id"][:, None].expand_as(anchor_index)
        anchor_summary = self._semantic_summary(anchor_semantic, anchor_label_id)
        answer_summary = self._semantic_summary(answer_semantic, answer_label_id)
        anchor_label = self.label_embedding(anchor_label_id.long())
        answer_label = self.label_embedding(answer_label_id.long())
        relation = self.relation_embedding(relation_id.long())
        pair_input = torch.cat(
            (
                batch["candidate_numeric"],
                anchor_rep,
                answer_rep,
                torch.abs(anchor_rep - answer_rep),
                anchor_rep * answer_rep,
                anchor_label,
                answer_label,
                relation,
                anchor_summary,
                answer_summary,
            ),
            dim=-1,
        )
        pair_score = self.pair_mlp(pair_input).squeeze(-1)
        none_input = torch.cat(
            (
                batch["candidate_numeric"],
                anchor_rep,
                anchor_label,
                relation,
                anchor_summary,
            ),
            dim=-1,
        )
        none_score = self.none_mlp(none_input).squeeze(-1)
        score = torch.where(batch["candidate_is_none"], none_score, pair_score)
        score = score.masked_fill(~batch["candidate_mask"], -1e4)
        proposal_own_logit = proposal_semantic.gather(
            -1, batch["proposal_label_id"].long()[..., None]
        ).squeeze(-1)
        return score, {
            "proposal_semantic_logits": proposal_semantic,
            "proposal_own_logit": proposal_own_logit,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal-checkpoint",
        type=Path,
        default=DEFAULT_BASE
        / "overlap_aware_red_epn_v2/overlap_aware_red_epn_v2_best.pt",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=DEFAULT_BASE / "pretrainedsed_beats_qces_detector.pt",
    )
    parser.add_argument(
        "--train-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_train/index.json",
    )
    parser.add_argument(
        "--dev-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_dev/index.json",
    )
    parser.add_argument(
        "--train-scenes", type=Path, default=DEFAULT_DATA / "scene_ids_overlap_train.txt"
    )
    parser.add_argument(
        "--dev-scenes", type=Path, default=DEFAULT_DATA / "scene_ids_overlap_dev.txt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE / "question_conditioned_acoustic_pair_ranker_v2",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4271)
    parser.add_argument("--max-proposals", type=int, default=120)
    parser.add_argument("--max-pairs", type=int, default=360)
    parser.add_argument("--max-anchors", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--proposal-batch-size", type=int, default=16)
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    parser.add_argument("--rank-batch-size", type=int, default=32)
    parser.add_argument("--semantic-epochs", type=int, default=18)
    parser.add_argument("--semantic-patience", type=int, default=5)
    parser.add_argument("--rank-epochs", type=int, default=30)
    parser.add_argument("--rank-patience", type=int, default=6)
    parser.add_argument("--semantic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--acoustic-finetune-learning-rate", type=float, default=2e-5)
    parser.add_argument("--rank-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-evidence-loss-weight", type=float, default=0.5)
    parser.add_argument("--validity-loss-weight", type=float, default=0.05)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bounds(start: float, end: float, frames: int) -> tuple[int, int]:
    left = max(0, min(frames - 1, int(math.floor(start * frames))))
    right = max(left + 1, min(frames, int(math.ceil(end * frames))))
    return left, right


@torch.inference_mode()
def collect_acoustic_scenes(
    model: ClassAwareRedEpnV1,
    store: DenseFeatureStore,
    scene_ids: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    max_proposals: int,
    max_scenes: int,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    dataset = ClassAwareSceneDataset(
        store, scene_ids, preload=True, max_scenes=max_scenes
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_class_aware,
    )
    rows: list[dict[str, Any]] = []
    model.eval()
    for raw in loader:
        outputs = model(
            raw["features"].to(device, non_blocking=True),
            raw["detector_logits"].to(device, non_blocking=True),
            raw["valid_mask"].to(device, non_blocking=True),
        )
        for batch_index, scene_id in enumerate(raw["scene_id"]):
            proposals = decode_class_aware_proposals(
                outputs,
                batch_index,
                max_classes=max_proposals,
                max_events=max_proposals,
            )
            events = list(raw["gold_events"][batch_index])
            proposal_valid = []
            for proposal in proposals:
                proposal_valid.append(
                    any(
                        int(proposal["label_id"]) == int(event["label_id"])
                        and interval_iou(
                            (float(proposal["start"]), float(proposal["end"])),
                            (float(event["start"]), float(event["end"])),
                        )
                        >= iou_threshold
                        for event in events
                    )
                )
            rows.append(
                {
                    "scene_id": str(scene_id),
                    "features": raw["features"][batch_index].half().clone(),
                    "events": events,
                    "proposals": proposals,
                    "proposal_valid": proposal_valid,
                }
            )
        if len(rows) % 256 < batch_size:
            print(f"candidate inference: {len(rows)}/{len(dataset)} scenes", flush=True)
    return rows


def pad_scene_bank(
    scene: Mapping[str, Any], config: AcousticRankerConfig
) -> dict[str, torch.Tensor]:
    proposals = list(scene["proposals"])
    events = list(scene["events"])
    result = {
        "scene_features": scene["features"],
        "proposal_start": torch.zeros(config.max_proposals),
        "proposal_end": torch.zeros(config.max_proposals),
        "proposal_label_id": torch.zeros(config.max_proposals, dtype=torch.long),
        "proposal_mask": torch.zeros(config.max_proposals, dtype=torch.bool),
        "proposal_valid_target": torch.zeros(config.max_proposals),
        "gold_start": torch.zeros(MAX_EVENTS_PER_SCENE),
        "gold_end": torch.zeros(MAX_EVENTS_PER_SCENE),
        "gold_label_id": torch.zeros(MAX_EVENTS_PER_SCENE, dtype=torch.long),
        "gold_mask": torch.zeros(MAX_EVENTS_PER_SCENE, dtype=torch.bool),
    }
    for index, proposal in enumerate(proposals[: config.max_proposals]):
        result["proposal_start"][index] = float(proposal["start"])
        result["proposal_end"][index] = float(proposal["end"])
        result["proposal_label_id"][index] = int(proposal["label_id"])
        result["proposal_mask"][index] = True
        result["proposal_valid_target"][index] = float(scene["proposal_valid"][index])
    if len(events) > MAX_EVENTS_PER_SCENE:
        raise ValueError(f"scene has {len(events)} events; increase MAX_EVENTS_PER_SCENE")
    for index, event in enumerate(events):
        result["gold_start"][index] = float(event["start"])
        result["gold_end"][index] = float(event["end"])
        result["gold_label_id"][index] = int(event["label_id"])
        result["gold_mask"][index] = True
    return result


def pair_numeric(
    anchor: Mapping[str, Any],
    answer: Mapping[str, Any] | None,
    *,
    relation: int,
    anchor_rank: int,
    answer_rank: int,
    temporal_rank: int,
    max_proposals: int,
) -> list[float]:
    anchor_start, anchor_end = float(anchor["start"]), float(anchor["end"])
    if answer is None:
        answer_start = answer_end = answer_score = answer_clip = 0.0
        same_label = 0.0
    else:
        answer_start, answer_end = float(answer["start"]), float(answer["end"])
        answer_score = float(answer["score"])
        answer_clip = float(answer.get("clip_score", answer_score))
        same_label = float(int(answer["label_id"]) == int(anchor["label_id"]))
    raw_delta = answer_start - anchor_start
    oriented = raw_delta if relation == 1 else -raw_delta
    relation_valid = float(answer is not None and oriented > 1e-6)
    gap = max(0.0, answer_start - anchor_end) if relation == 1 else max(
        0.0, anchor_start - answer_end
    )
    overlap = 0.0
    if answer is not None:
        intersection = max(0.0, min(anchor_end, answer_end) - max(anchor_start, answer_start))
        overlap = intersection / max(
            min(anchor_end - anchor_start, answer_end - answer_start), 1e-8
        )
    return [
        float(anchor["score"]),
        float(anchor.get("clip_score", anchor["score"])),
        anchor_end - anchor_start,
        anchor_rank / max(max_proposals - 1, 1),
        answer_score,
        answer_clip,
        answer_end - answer_start if answer is not None else 0.0,
        answer_rank / max(max_proposals - 1, 1),
        raw_delta,
        oriented,
        abs(raw_delta),
        gap,
        overlap,
        relation_valid,
        1.0 / (1.0 + temporal_rank),
        anchor_start,
        anchor_end,
        answer_start,
        answer_end,
        same_label,
    ]


def build_question_examples(
    scenes: Sequence[Mapping[str, Any]],
    config: AcousticRankerConfig,
    *,
    iou_threshold: float,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int]]:
    examples: list[dict[str, torch.Tensor]] = []
    counts: Counter[str] = Counter()
    for scene_index, scene in enumerate(scenes):
        events = list(scene["events"])
        proposals = list(scene["proposals"])
        pairs = adjacent_index_pairs(events)
        if len(pairs) != max(len(events) - 1, 0):
            counts["scenes_skipped_tied_onset"] += 1
            continue
        qa_specs: list[tuple[int, int | None, int]] = []
        for left, right in pairs:
            qa_specs.extend(((left, right, 1), (right, left, 0)))
        order = sorted(
            range(len(events)), key=lambda index: (events[index]["start"], events[index]["end"])
        )
        if order:
            qa_specs.extend(((order[0], None, 0), (order[-1], None, 1)))

        for anchor_gold_index, answer_gold_index, relation in qa_specs:
            anchor_gold = events[anchor_gold_index]
            answer_gold = events[answer_gold_index] if answer_gold_index is not None else None
            anchor_indices = [
                index
                for index, proposal in enumerate(proposals)
                if int(proposal["label_id"]) == int(anchor_gold["label_id"])
            ][: config.max_anchors]
            pair_rows = [
                (anchor_index, answer_index)
                for anchor_index in anchor_indices
                for answer_index, answer in enumerate(proposals)
                if int(answer["label_id"])
                != int(proposals[anchor_index]["label_id"])
            ]
            # Keep the broad proposal pool but make truncation deterministic if
            # future decoders emit more than three anchor occurrences.
            pair_rows.sort(
                key=lambda item: (
                    item[0] + item[1],
                    abs(
                        float(proposals[item[1]]["start"])
                        - float(proposals[item[0]]["start"])
                    ),
                )
            )
            pair_rows = pair_rows[: config.max_pairs]
            candidate_rows: list[tuple[int, int, bool]] = [
                (anchor_index, answer_index, False)
                for anchor_index, answer_index in pair_rows
            ]
            candidate_rows.extend((anchor_index, -1, True) for anchor_index in anchor_indices)
            if len(candidate_rows) > config.list_size:
                raise RuntimeError("candidate list exceeds configured list size")

            candidate_numeric = torch.zeros(
                config.list_size, config.numeric_dim, dtype=torch.float16
            )
            pair_anchor_index = torch.zeros(config.list_size, dtype=torch.int16)
            pair_answer_index = torch.full(
                (config.list_size,), -1, dtype=torch.int16
            )
            candidate_mask = torch.zeros(config.list_size, dtype=torch.bool)
            candidate_is_none = torch.zeros(config.list_size, dtype=torch.bool)
            candidate_anchor_start = torch.zeros(config.list_size)
            candidate_anchor_end = torch.zeros(config.list_size)
            candidate_answer_start = torch.zeros(config.list_size)
            candidate_answer_end = torch.zeros(config.list_size)
            candidate_answer_label = torch.full(
                (config.list_size,), -1, dtype=torch.long
            )
            target = torch.zeros(config.list_size)
            positive: list[tuple[int, float]] = []
            temporal_order = sorted(
                range(len(pair_rows)),
                key=lambda index: abs(
                    float(proposals[pair_rows[index][1]]["start"])
                    - float(proposals[pair_rows[index][0]]["start"])
                ),
            )
            temporal_rank = {value: rank for rank, value in enumerate(temporal_order)}
            for index, (anchor_index, answer_index, is_none) in enumerate(candidate_rows):
                anchor = proposals[anchor_index]
                answer = None if is_none else proposals[answer_index]
                candidate_numeric[index] = torch.tensor(
                    pair_numeric(
                        anchor,
                        answer,
                        relation=relation,
                        anchor_rank=anchor_index,
                        answer_rank=max(answer_index, 0),
                        temporal_rank=temporal_rank.get(index, len(pair_rows)),
                        max_proposals=config.max_proposals,
                    ),
                    dtype=torch.float16,
                )
                pair_anchor_index[index] = anchor_index
                pair_answer_index[index] = answer_index
                candidate_mask[index] = True
                candidate_is_none[index] = is_none
                candidate_anchor_start[index] = float(anchor["start"])
                candidate_anchor_end[index] = float(anchor["end"])
                anchor_iou = interval_iou(
                    (float(anchor["start"]), float(anchor["end"])),
                    (float(anchor_gold["start"]), float(anchor_gold["end"])),
                )
                if is_none:
                    if answer_gold is None and anchor_iou >= iou_threshold:
                        positive.append((index, math.exp(4.0 * anchor_iou)))
                    continue
                assert answer is not None
                candidate_answer_start[index] = float(answer["start"])
                candidate_answer_end[index] = float(answer["end"])
                candidate_answer_label[index] = int(answer["label_id"])
                if answer_gold is None:
                    continue
                answer_iou = interval_iou(
                    (float(answer["start"]), float(answer["end"])),
                    (float(answer_gold["start"]), float(answer_gold["end"])),
                )
                if (
                    anchor_iou >= iou_threshold
                    and int(answer["label_id"]) == int(answer_gold["label_id"])
                    and answer_iou >= iou_threshold
                ):
                    positive.append(
                        (index, math.exp(2.0 * (anchor_iou + answer_iou)))
                    )
            if positive:
                denominator = sum(weight for _, weight in positive)
                for index, weight in positive:
                    target[index] = weight / denominator

            no_evidence = answer_gold is None
            counts["questions"] += 1
            counts["no_evidence" if no_evidence else "answerable"] += 1
            counts["anchor_candidate_found"] += int(bool(anchor_indices))
            counts["target_supported"] += int(bool(positive))
            if not no_evidence:
                counts[
                    "answerable_supported" if positive else "answerable_unsupported"
                ] += 1
            else:
                counts[
                    "no_evidence_supported" if positive else "no_evidence_unsupported"
                ] += 1
            examples.append(
                {
                    "scene_index": torch.tensor(scene_index),
                    "candidate_numeric": candidate_numeric,
                    "pair_anchor_index": pair_anchor_index,
                    "pair_answer_index": pair_answer_index,
                    "candidate_mask": candidate_mask,
                    "candidate_is_none": candidate_is_none,
                    "candidate_anchor_start": candidate_anchor_start,
                    "candidate_anchor_end": candidate_anchor_end,
                    "candidate_answer_start": candidate_answer_start,
                    "candidate_answer_end": candidate_answer_end,
                    "candidate_answer_label": candidate_answer_label,
                    "anchor_label_id": torch.tensor(int(anchor_gold["label_id"])),
                    "relation_id": torch.tensor(relation),
                    "target": target,
                    "target_available": torch.tensor(bool(positive)),
                    "no_evidence": torch.tensor(no_evidence),
                    "gold_anchor_start": torch.tensor(float(anchor_gold["start"])),
                    "gold_anchor_end": torch.tensor(float(anchor_gold["end"])),
                    "gold_answer_label": torch.tensor(
                        -1 if answer_gold is None else int(answer_gold["label_id"])
                    ),
                    "gold_answer_start": torch.tensor(
                        0.0 if answer_gold is None else float(answer_gold["start"])
                    ),
                    "gold_answer_end": torch.tensor(
                        0.0 if answer_gold is None else float(answer_gold["end"])
                    ),
                }
            )
    return examples, {key: int(value) for key, value in counts.items()}


class SemanticSceneDataset(Dataset[Mapping[str, torch.Tensor]]):
    def __init__(self, banks: Sequence[Mapping[str, torch.Tensor]]) -> None:
        self.banks = list(banks)

    def __len__(self) -> int:
        return len(self.banks)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        bank = self.banks[index]
        return {
            key: bank[key]
            for key in (
                "scene_features",
                "gold_start",
                "gold_end",
                "gold_label_id",
                "gold_mask",
            )
        }


class QuestionDataset(Dataset[Mapping[str, torch.Tensor]]):
    def __init__(
        self,
        banks: Sequence[Mapping[str, torch.Tensor]],
        examples: Sequence[Mapping[str, torch.Tensor]],
    ) -> None:
        self.banks = list(banks)
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        example = self.examples[index]
        bank = self.banks[int(example["scene_index"])]
        return dict(example) | {
            key: bank[key]
            for key in (
                "scene_features",
                "proposal_start",
                "proposal_end",
                "proposal_label_id",
                "proposal_mask",
                "proposal_valid_target",
            )
        }


def to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    result = {
        key: value.to(device, non_blocking=True) for key, value in batch.items()
    }
    for key in ("scene_features", "candidate_numeric"):
        if key in result:
            result[key] = result[key].float()
    return result


def class_weights_from_scenes(
    scenes: Sequence[Mapping[str, Any]], num_classes: int, device: torch.device
) -> torch.Tensor:
    count = torch.zeros(num_classes)
    for scene in scenes:
        for event in scene["events"]:
            count[int(event["label_id"])] += 1
    weight = count.clamp_min(1).rsqrt()
    return (weight / weight.mean()).clamp(0.5, 2.0).to(device)


@torch.inference_mode()
def evaluate_semantic(
    model: AcousticPairNoneRanker,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    scores = []
    targets = []
    for raw in loader:
        batch = to_device(raw, device)
        _, semantic = model.encode_spans(
            batch["scene_features"],
            batch["gold_start"],
            batch["gold_end"],
            batch["gold_mask"],
        )
        selected = batch["gold_mask"]
        scores.append(semantic[selected].float().cpu())
        targets.append(batch["gold_label_id"][selected].cpu())
    score = torch.cat(scores)
    target = torch.cat(targets)
    order = score.argsort(dim=1, descending=True)
    result: dict[str, float | int] = {"events": int(target.numel())}
    for k in (1, 5, 20):
        result[f"oracle_span_top{k}_accuracy_↑"] = float(
            (order[:, :k] == target[:, None]).any(dim=1).float().mean()
        )
    return result


def normalized_binary_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    positive = mask & (target > 0.5)
    negative = mask & ~positive
    positive_loss = (
        F.softplus(-logits[positive]).mean()
        if positive.any()
        else logits.sum() * 0.0
    )
    negative_loss = (
        F.softplus(logits[negative]).mean()
        if negative.any()
        else logits.sum() * 0.0
    )
    return 0.6 * positive_loss + 0.4 * negative_loss


@torch.inference_mode()
def collect_rank_outputs(
    model: AcousticPairNoneRanker,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    keys = (
        "candidate_mask",
        "candidate_is_none",
        "candidate_anchor_start",
        "candidate_anchor_end",
        "candidate_answer_start",
        "candidate_answer_end",
        "candidate_answer_label",
        "no_evidence",
        "gold_anchor_start",
        "gold_anchor_end",
        "gold_answer_label",
        "gold_answer_start",
        "gold_answer_end",
    )
    result: dict[str, list[torch.Tensor]] = {"scores": []} | {
        key: [] for key in keys
    }
    for raw in loader:
        batch = to_device(raw, device)
        score, _ = model(batch)
        result["scores"].append(score.float().cpu())
        for key in keys:
            result[key].append(raw[key].cpu())
    return {key: torch.cat(value) for key, value in result.items()}


def metrics_from_outputs(
    outputs: Mapping[str, torch.Tensor],
    *,
    none_bias: float,
    iou_threshold: float,
) -> dict[str, float | int]:
    score = outputs["scores"].clone()
    score += outputs["candidate_is_none"].to(score.dtype) * none_bias
    prediction = score.argmax(dim=1)
    counts: Counter[str] = Counter()
    for row, choice_tensor in enumerate(prediction):
        choice = int(choice_tensor)
        no_evidence = bool(outputs["no_evidence"][row])
        predicted_none = bool(outputs["candidate_is_none"][row, choice])
        anchor_iou = interval_iou(
            (
                float(outputs["candidate_anchor_start"][row, choice]),
                float(outputs["candidate_anchor_end"][row, choice]),
            ),
            (
                float(outputs["gold_anchor_start"][row]),
                float(outputs["gold_anchor_end"][row]),
            ),
        )
        counts["questions"] += 1
        if no_evidence:
            counts["no_evidence"] += 1
            counts["no_evidence_correct"] += int(predicted_none)
            counts["verified_no_evidence_correct"] += int(
                predicted_none and anchor_iou >= iou_threshold
            )
            continue
        counts["answerable"] += 1
        if predicted_none:
            continue
        label_correct = int(
            int(outputs["candidate_answer_label"][row, choice])
            == int(outputs["gold_answer_label"][row])
        )
        answer_iou = interval_iou(
            (
                float(outputs["candidate_answer_start"][row, choice]),
                float(outputs["candidate_answer_end"][row, choice]),
            ),
            (
                float(outputs["gold_answer_start"][row]),
                float(outputs["gold_answer_end"][row]),
            ),
        )
        answer_correct = label_correct and answer_iou >= iou_threshold
        counts["answer_label_correct"] += label_correct
        counts["answer_event_correct"] += int(answer_correct)
        counts["anchor_correct"] += int(anchor_iou >= iou_threshold)
        counts["joint_evidence_correct"] += int(
            answer_correct and anchor_iou >= iou_threshold
        )
    answerable = max(counts["answerable"], 1)
    no_evidence_count = max(counts["no_evidence"], 1)
    joint = counts["joint_evidence_correct"] / answerable
    verified = counts["verified_no_evidence_correct"] / no_evidence_count
    harmonic = 0.0 if joint + verified <= 0 else 2.0 * joint * verified / (joint + verified)
    return {
        "questions": int(counts["questions"]),
        "answerable": int(counts["answerable"]),
        "no_evidence": int(counts["no_evidence"]),
        "answerable_label_accuracy_↑": counts["answer_label_correct"] / answerable,
        "answerable_anchor_iou030_accuracy_↑": counts["anchor_correct"] / answerable,
        "answerable_answer_event_iou030_accuracy_↑": counts["answer_event_correct"]
        / answerable,
        "answerable_joint_evidence_iou030_accuracy_↑": joint,
        "no_evidence_accuracy_↑": counts["no_evidence_correct"] / no_evidence_count,
        "verified_no_evidence_iou030_accuracy_↑": verified,
        "balanced_joint_verified_accuracy_↑": 0.5 * (joint + verified),
        "harmonic_joint_verified_accuracy_↑": harmonic,
        "none_bias": none_bias,
    }


def calibrate_none_bias(
    outputs: Mapping[str, torch.Tensor], *, iou_threshold: float
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    best_key: tuple[float, ...] | None = None
    for none_bias in np.linspace(-4.0, 3.0, 57):
        metrics = metrics_from_outputs(
            outputs, none_bias=float(none_bias), iou_threshold=iou_threshold
        )
        key = (
            float(metrics["harmonic_joint_verified_accuracy_↑"]),
            min(
                float(metrics["answerable_joint_evidence_iou030_accuracy_↑"]),
                float(metrics["verified_no_evidence_iou030_accuracy_↑"]),
            ),
            float(metrics["answerable_label_accuracy_↑"]),
        )
        if best_key is None or key > best_key:
            best_key, best = key, metrics
    if best is None:
        raise RuntimeError("NONE calibration produced no metrics")
    return best


def main() -> None:
    args = parse_args()
    if args.max_pairs < 1 or args.max_anchors < 1 or args.max_proposals < 2:
        raise ValueError("invalid candidate limits")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = _device(args.device)

    proposal_payload = torch.load(
        args.proposal_checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    proposal_config = ClassAwareRedEpnV1Config(**dict(proposal_payload["config"]))
    proposal_model = ClassAwareRedEpnV1(proposal_config)
    proposal_model.load_state_dict(proposal_payload["model_state_dict"], strict=True)
    proposal_model.to(device).eval()
    store = DenseFeatureStore(
        [args.train_index.resolve(), args.dev_index.resolve()], cache_size=32
    )
    split_ids = {
        "train": load_scene_list(args.train_scenes.resolve()),
        "dev": load_scene_list(args.dev_scenes.resolve()),
    }
    scenes = {
        split: collect_acoustic_scenes(
            proposal_model,
            store,
            ids,
            device=device,
            batch_size=args.proposal_batch_size,
            max_proposals=args.max_proposals,
            max_scenes=args.max_train_scenes if split == "train" else args.max_dev_scenes,
            iou_threshold=args.iou_threshold,
        )
        for split, ids in split_ids.items()
    }
    del proposal_model
    torch.cuda.empty_cache()
    config = AcousticRankerConfig(
        feature_dim=proposal_config.feature_dim,
        num_classes=proposal_config.num_classes,
        max_proposals=args.max_proposals,
        max_pairs=args.max_pairs,
        max_anchors=args.max_anchors,
    )
    banks = {
        split: [pad_scene_bank(scene, config) for scene in rows]
        for split, rows in scenes.items()
    }
    examples_and_counts = {
        split: build_question_examples(
            rows, config, iou_threshold=args.iou_threshold
        )
        for split, rows in scenes.items()
    }
    question_datasets = {
        split: QuestionDataset(banks[split], examples_and_counts[split][0])
        for split in scenes
    }
    semantic_datasets = {
        split: SemanticSceneDataset(banks[split]) for split in scenes
    }
    loader_common = {
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    semantic_loaders = {
        "train": DataLoader(
            semantic_datasets["train"],
            batch_size=args.semantic_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            **loader_common,
        ),
        "dev": DataLoader(
            semantic_datasets["dev"],
            batch_size=args.semantic_batch_size,
            shuffle=False,
            **loader_common,
        ),
    }
    train_indices = [
        index
        for index, row in enumerate(question_datasets["train"].examples)
        if bool(row["target_available"])
    ]
    rank_loaders = {
        "train": DataLoader(
            Subset(question_datasets["train"], train_indices),
            batch_size=args.rank_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + 1),
            **loader_common,
        ),
        "dev": DataLoader(
            question_datasets["dev"],
            batch_size=args.rank_batch_size,
            shuffle=False,
            **loader_common,
        ),
    }

    model = AcousticPairNoneRanker(config).to(device)
    class_weight = class_weights_from_scenes(
        scenes["train"], config.num_classes, device
    )
    semantic_optimizer = torch.optim.AdamW(
        list(model.frame_encoder.parameters())
        + list(model.frame_attention.parameters())
        + list(model.span_projection.parameters())
        + list(model.semantic_classifier.parameters()),
        lr=args.semantic_learning_rate,
        weight_decay=args.weight_decay,
    )
    semantic_history: list[dict[str, Any]] = []
    semantic_best_key: tuple[float, ...] | None = None
    semantic_best_epoch = 0
    semantic_best_metrics: Mapping[str, Any] | None = None
    semantic_best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, args.semantic_epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for raw in semantic_loaders["train"]:
            batch = to_device(raw, device)
            semantic_optimizer.zero_grad(set_to_none=True)
            _, logits = model.encode_spans(
                batch["scene_features"],
                batch["gold_start"],
                batch["gold_end"],
                batch["gold_mask"],
            )
            selected = batch["gold_mask"]
            loss = F.cross_entropy(
                logits[selected],
                batch["gold_label_id"][selected],
                weight=class_weight,
                label_smoothing=0.05,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            semantic_optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        metrics = evaluate_semantic(model, semantic_loaders["dev"], device)
        key = (
            float(metrics["oracle_span_top1_accuracy_↑"]),
            float(metrics["oracle_span_top5_accuracy_↑"]),
        )
        row = {
            "stage": "semantic",
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "dev": metrics,
        }
        semantic_history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if semantic_best_key is None or key > semantic_best_key:
            semantic_best_key = key
            semantic_best_epoch = epoch
            semantic_best_metrics = row
            semantic_best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.semantic_patience:
            print(f"semantic early stopping at epoch {epoch}", flush=True)
            break
    if semantic_best_state is None or semantic_best_metrics is None:
        raise RuntimeError("semantic stage produced no checkpoint")
    model.load_state_dict(semantic_best_state, strict=True)

    acoustic_parameters = (
        list(model.frame_encoder.parameters())
        + list(model.frame_attention.parameters())
        + list(model.span_projection.parameters())
        + list(model.semantic_classifier.parameters())
    )
    acoustic_ids = {id(parameter) for parameter in acoustic_parameters}
    rank_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in acoustic_ids
    ]
    rank_optimizer = torch.optim.AdamW(
        [
            {"params": acoustic_parameters, "lr": args.acoustic_finetune_learning_rate},
            {"params": rank_parameters, "lr": args.rank_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    rank_history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_metrics: Mapping[str, Any] | None = None
    stale = 0
    checkpoint_path = output_dir / "acoustic_pair_ranker_v2_best.pt"
    for epoch in range(1, args.rank_epochs + 1):
        model.train()
        total_loss = total_rank = total_validity = 0.0
        batches = 0
        for raw in rank_loaders["train"]:
            batch = to_device(raw, device)
            rank_optimizer.zero_grad(set_to_none=True)
            score, auxiliary = model(batch)
            log_probability = score.log_softmax(dim=1)
            per_example = -(batch["target"] * log_probability).sum(dim=1)
            weight = torch.where(
                batch["no_evidence"],
                torch.full_like(per_example, args.no_evidence_loss_weight),
                torch.ones_like(per_example),
            )
            rank_loss = (per_example * weight).sum() / weight.sum().clamp_min(1.0)
            validity_loss = normalized_binary_loss(
                auxiliary["proposal_own_logit"],
                batch["proposal_valid_target"],
                batch["proposal_mask"],
            )
            loss = rank_loss + args.validity_loss_weight * validity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            rank_optimizer.step()
            total_loss += float(loss.detach())
            total_rank += float(rank_loss.detach())
            total_validity += float(validity_loss.detach())
            batches += 1
        metrics = calibrate_none_bias(
            collect_rank_outputs(model, rank_loaders["dev"], device),
            iou_threshold=args.iou_threshold,
        )
        key = (
            float(metrics["harmonic_joint_verified_accuracy_↑"]),
            min(
                float(metrics["answerable_joint_evidence_iou030_accuracy_↑"]),
                float(metrics["verified_no_evidence_iou030_accuracy_↑"]),
            ),
            float(metrics["answerable_label_accuracy_↑"]),
        )
        row = {
            "stage": "rank",
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "train_rank_loss": total_rank / max(batches, 1),
            "train_validity_loss": total_validity / max(batches, 1),
            "dev": metrics,
            "selection_key": list(key),
        }
        rank_history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_metrics, stale = key, epoch, row, 0
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "selection_metrics": row,
                    "none_bias": float(metrics["none_bias"]),
                    "semantic_best_epoch": semantic_best_epoch,
                    "semantic_best_metrics": semantic_best_metrics,
                    "proposal_checkpoint": str(args.proposal_checkpoint.resolve()),
                    "proposal_checkpoint_sha256": _sha256_file(
                        args.proposal_checkpoint.resolve()
                    ),
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= args.rank_patience:
            print(f"rank early stopping at epoch {epoch}", flush=True)
            break
    if best_metrics is None:
        raise RuntimeError("rank stage produced no checkpoint")

    v1_receipt_path = DEFAULT_BASE / "listwise_pair_none_ranker_v1/receipt.json"
    v1_metrics = None
    if v1_receipt_path.is_file():
        v1_metrics = json.loads(v1_receipt_path.read_text(encoding="utf-8")).get(
            "best_metrics"
        )
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "frozen proposals + all anchor-answer pairs + BEATs span encoder + per-anchor NONE",
        "answer_label_used_as_model_input": False,
        "hard_relation_filter_used": False,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "config": asdict(config),
        "data": {
            split: {
                "scenes": len(scenes[split]),
                "questions": len(question_datasets[split]),
                "counts": examples_and_counts[split][1],
            }
            for split in scenes
        },
        "trainable_questions": len(train_indices),
        "semantic_best_epoch": semantic_best_epoch,
        "semantic_best_metrics": semantic_best_metrics,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "feature_only_v1_reference": v1_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "proposal_checkpoint": str(args.proposal_checkpoint.resolve()),
        "proposal_checkpoint_sha256": _sha256_file(args.proposal_checkpoint.resolve()),
        "semantic_history": semantic_history,
        "rank_history": rank_history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(
        json.dumps(
            {"complete": True, "best_epoch": best_epoch, "best_metrics": best_metrics},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
