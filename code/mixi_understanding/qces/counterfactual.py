"""Paired counterfactual supervision for QCES-v5.

The objectives in this module operate on records that have already passed
through the composer and separator.  They therefore do not call AudioSep a
second time: every record contributes one model output and paired losses are
computed by indexing those batched outputs.

These controls test evidence equivariance under declared dataset
interventions.  They are not, by themselves, proof of a causal mechanism.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.model import QCESOutput


SURFACE_LOSS_NAMES = (
    "surface_semantic_invariance",
    "surface_role_invariance",
    "surface_no_evidence_invariance",
    "surface_evidence_invariance",
)
FAMILY_LOSS_NAMES = (
    "family_temporal_delta",
    "family_evidence_delta",
    "family_no_evidence_transition",
)
QUESTION_LOSS_NAMES = (
    "question_temporal_delta",
    "question_evidence_delta",
)
COUNTERFACTUAL_LOSS_NAMES = (
    *SURFACE_LOSS_NAMES,
    *FAMILY_LOSS_NAMES,
    *QUESTION_LOSS_NAMES,
)

# Every metric is explicitly labelled because the project reports experimental
# tables to readers who should not need to infer whether larger is better.
COUNTERFACTUAL_METRIC_DIRECTIONS: Dict[str, Dict[str, str]] = {
    "surface_semantic_cosine": {
        "direction": "maximize",
        "display": "surface semantic cosine ↑",
    },
    "surface_role_probability_mae": {
        "direction": "minimize",
        "display": "surface role-probability MAE ↓",
    },
    "surface_no_evidence_probability_mae": {
        "direction": "minimize",
        "display": "surface no-evidence probability MAE ↓",
    },
    "surface_evidence_relative_l1": {
        "direction": "minimize",
        "display": "surface evidence relative L1 ↓",
    },
    "family_temporal_delta_rmse": {
        "direction": "minimize",
        "display": "family temporal-delta RMSE ↓",
    },
    "family_temporal_delta_cosine": {
        "direction": "maximize",
        "display": "family temporal-delta cosine ↑",
    },
    "family_evidence_delta_relative_l1": {
        "direction": "minimize",
        "display": "family evidence-delta relative L1 ↓",
    },
    "family_evidence_delta_cosine": {
        "direction": "maximize",
        "display": "family evidence-delta cosine ↑",
    },
    "family_transition_accuracy": {
        "direction": "maximize",
        "display": "primary A→B→no-evidence transition accuracy ↑",
    },
    "question_temporal_delta_rmse": {
        "direction": "minimize",
        "display": "question temporal-delta RMSE ↓",
    },
    "question_temporal_delta_cosine": {
        "direction": "maximize",
        "display": "question temporal-delta cosine ↑",
    },
    "question_evidence_delta_relative_l1": {
        "direction": "minimize",
        "display": "question evidence-delta relative L1 ↓",
    },
    "question_evidence_delta_cosine": {
        "direction": "maximize",
        "display": "question evidence-delta cosine ↑",
    },
}


@dataclass(frozen=True)
class CounterfactualGroupPlan:
    """A deterministic, disjoint cover of records with required paired blocks."""

    sample_ids: tuple[str, ...]
    surface_pairs: tuple[tuple[int, int], ...]
    primary_triplets: tuple[tuple[int, int, int], ...]
    question_pairs: tuple[tuple[int, int], ...]
    blocks: tuple[tuple[int, ...], ...]
    surface_enabled: bool
    family_enabled: bool
    question_enabled: bool

    def __post_init__(self) -> None:
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("counterfactual plan requires unique sample IDs")
        flattened = [index for block in self.blocks for index in block]
        if sorted(flattened) != list(range(len(self.sample_ids))):
            raise ValueError(
                "counterfactual blocks must cover every record exactly once"
            )

    def batch_groups(self, batch_sample_ids: Sequence[str]) -> Dict[str, Any]:
        """Translate complete global groups into positions in one collated batch."""

        if len(batch_sample_ids) != len(set(batch_sample_ids)):
            raise ValueError("a paired batch cannot repeat a sample ID")
        unknown = sorted(set(batch_sample_ids) - set(self.sample_ids))
        if unknown:
            raise ValueError(
                f"paired batch contains IDs absent from its group plan: {unknown[:5]}"
            )
        position = {sample_id: index for index, sample_id in enumerate(batch_sample_ids)}

        def present(groups: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
            result = []
            for group in groups:
                ids = tuple(self.sample_ids[index] for index in group)
                if all(sample_id in position for sample_id in ids):
                    result.append(tuple(position[sample_id] for sample_id in ids))
            return result

        surface = present(self.surface_pairs)
        primary = present(self.primary_triplets)
        question = present(self.question_pairs)
        return {
            "surface_pairs": surface,
            # Primary triplets are always ordered base, order_swap, anchor_drop.
            "primary_triplets": primary,
            "question_pairs": question,
            "plan_enabled": {
                "surface": self.surface_enabled,
                "family": self.family_enabled,
                "question": self.question_enabled,
            },
            "counts": {
                "surface": len(surface),
                "family": len(primary),
                "question": len(question),
            },
        }

    def provenance(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "surface_invariance": self.surface_enabled,
            "family_equivariance": self.family_enabled,
            "question_equivariance": self.question_enabled,
            "surface_pair_count": len(self.surface_pairs),
            "primary_triplet_count": len(self.primary_triplets),
            "question_pair_count": len(self.question_pairs),
            "record_count": len(self.sample_ids),
            "record_coverage_fraction ↑": 1.0,
            "duplicate_record_calls_per_epoch ↓": 0,
            "warning": (
                "Declared intervention equivariance is evaluated; this is not "
                "a claim of causal identification or causal proof."
            ),
            "metric_directions": COUNTERFACTUAL_METRIC_DIRECTIONS,
        }


def _record_target_signature(record: QCESV5Record) -> tuple[Any, ...]:
    return (
        bool(record.no_evidence),
        tuple(sorted(record.evidence_event_ids)),
    )


def _require_v5(records: Sequence[Any]) -> tuple[QCESV5Record, ...]:
    if not records or not all(isinstance(record, QCESV5Record) for record in records):
        raise ValueError(
            "paired counterfactual objectives require a complete QCES-v5 manifest"
        )
    typed = tuple(records)
    splits = {record.split for record in typed}
    if len(splits) != 1:
        raise ValueError(
            f"counterfactual groups cannot cross dataset splits: {sorted(splits)}"
        )
    return typed


def build_counterfactual_group_plan(
    records: Sequence[Any],
    *,
    enable_surface: bool,
    enable_family: bool,
    enable_question: bool,
) -> CounterfactualGroupPlan:
    """Validate requested groups and build a no-duplication epoch cover.

    Surface pairs and primary triples come from explicit v5 group IDs.  The
    optional same-scene/different-question pairs are a deterministic matching
    over records not already consumed by those stronger controls.  This keeps
    one separator call per record per epoch while retaining paired batches.
    """

    if not (enable_surface or enable_family or enable_question):
        raise ValueError("at least one counterfactual group type must be enabled")
    typed_records = _require_v5(records)
    sample_ids = tuple(record.sample_id for record in typed_records)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("counterfactual training requires unique sample IDs")
    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    occupied: set[int] = set()
    surface_pairs: list[tuple[int, int]] = []
    primary_triplets: list[tuple[int, int, int]] = []
    question_pairs: list[tuple[int, int]] = []

    if enable_surface:
        surface_groups: Dict[str, list[QCESV5Record]] = defaultdict(list)
        first_records = [record for record in typed_records if record.relation == "first"]
        for record in first_records:
            if record.surface_control_group_id is None:
                raise ValueError(
                    f"incomplete surface control: {record.sample_id} has no group ID"
                )
            surface_groups[record.surface_control_group_id].append(record)
        if not surface_groups:
            raise ValueError("surface invariance requested but no surface pairs exist")
        for group_id in sorted(surface_groups):
            pair = surface_groups[group_id]
            variants = {record.mention_order_variant for record in pair}
            if len(pair) != 2 or variants != {"forward", "reversed"}:
                raise ValueError(f"incomplete surface control group: {group_id}")
            by_order = {record.mention_order_variant: record for record in pair}
            forward = by_order["forward"]
            reversed_record = by_order["reversed"]
            invariant = lambda record: (
                record.scene_id,
                record.variant_id,
                record.question_semantics_id,
                record.answer,
                record.no_evidence,
                tuple(record.anchor_event_ids),
                tuple(record.answer_event_ids),
                tuple(record.evidence_event_ids),
                record.mixture_path,
            )
            if invariant(forward) != invariant(reversed_record):
                raise ValueError(f"surface control changes audio semantics: {group_id}")
            indices = (
                index_by_id[forward.sample_id],
                index_by_id[reversed_record.sample_id],
            )
            if occupied.intersection(indices):
                raise ValueError(f"overlapping surface control group: {group_id}")
            surface_pairs.append(indices)
            occupied.update(indices)

    if enable_family:
        primary_groups: Dict[str, list[QCESV5Record]] = defaultdict(list)
        for record in typed_records:
            if record.primary_counterfactual_probe:
                primary_groups[record.counterfactual_group_id].append(record)
        if not primary_groups:
            raise ValueError("family equivariance requested but no primary triples exist")
        expected_variants = {"base", "order_swap", "anchor_drop"}
        for group_id in sorted(primary_groups):
            group = primary_groups[group_id]
            if len(group) != 3 or {record.variant_id for record in group} != expected_variants:
                raise ValueError(f"incomplete primary counterfactual group: {group_id}")
            by_variant = {record.variant_id: record for record in group}
            base = by_variant["base"]
            swap = by_variant["order_swap"]
            drop = by_variant["anchor_drop"]
            shared = lambda record: (
                record.scene_family_id,
                record.question_semantics_id,
                record.question,
                tuple(record.answer_options),
            )
            if len({shared(base), shared(swap), shared(drop)}) != 1:
                raise ValueError(f"primary counterfactual surface changed: {group_id}")
            if base.no_evidence or swap.no_evidence or not drop.no_evidence:
                raise ValueError(f"primary group lacks A→B→no-evidence states: {group_id}")
            if base.answer == swap.answer or drop.answer == base.answer:
                raise ValueError(f"primary group lacks distinct A/B/no-evidence answers: {group_id}")
            indices = (
                index_by_id[base.sample_id],
                index_by_id[swap.sample_id],
                index_by_id[drop.sample_id],
            )
            if occupied.intersection(indices):
                raise ValueError(f"paired controls overlap primary group: {group_id}")
            primary_triplets.append(indices)
            occupied.update(indices)

    if enable_question:
        records_by_scene: Dict[str, list[QCESV5Record]] = defaultdict(list)
        for index, record in enumerate(typed_records):
            if index not in occupied:
                records_by_scene[record.scene_id].append(record)
        if not records_by_scene:
            raise ValueError(
                "question equivariance requested but stronger controls consume all records"
            )
        for scene_id in sorted(records_by_scene):
            available = sorted(
                records_by_scene[scene_id],
                key=lambda record: (record.question_semantics_id, record.sample_id),
            )
            made_pair = False
            while len(available) >= 2:
                left = available.pop(0)
                partner_index = next(
                    (
                        index
                        for index, candidate in enumerate(available)
                        if candidate.question_semantics_id != left.question_semantics_id
                        and _record_target_signature(candidate)
                        != _record_target_signature(left)
                    ),
                    None,
                )
                if partner_index is None:
                    continue
                right = available.pop(partner_index)
                indices = (
                    index_by_id[left.sample_id],
                    index_by_id[right.sample_id],
                )
                question_pairs.append(indices)
                occupied.update(indices)
                made_pair = True
            if not made_pair:
                raise ValueError(
                    "incomplete same-scene/different-question controls for "
                    f"scene {scene_id}"
                )
        if not question_pairs:
            raise ValueError(
                "question equivariance requested but no non-trivial oracle deltas exist"
            )

    grouped_blocks: list[tuple[int, ...]] = [
        *surface_pairs,
        *primary_triplets,
        *question_pairs,
    ]
    covered = {index for block in grouped_blocks for index in block}
    if len(covered) != sum(len(block) for block in grouped_blocks):
        raise ValueError("counterfactual group construction duplicated a record")
    singleton_blocks = [
        (index,) for index in range(len(typed_records)) if index not in covered
    ]
    blocks = tuple((*grouped_blocks, *singleton_blocks))
    return CounterfactualGroupPlan(
        sample_ids=sample_ids,
        surface_pairs=tuple(surface_pairs),
        primary_triplets=tuple(primary_triplets),
        question_pairs=tuple(question_pairs),
        blocks=blocks,
        surface_enabled=enable_surface,
        family_enabled=enable_family,
        question_enabled=enable_question,
    )


class CounterfactualBatchSampler(Sampler[list[int]]):
    """Pack validated groups without splitting or duplicating any record."""

    def __init__(
        self,
        plan: CounterfactualGroupPlan,
        batch_size: int,
        *,
        seed: int,
        shuffle: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        largest = max(len(block) for block in plan.blocks)
        if largest > batch_size:
            raise ValueError(
                f"batch_size={batch_size} cannot contain required paired group "
                f"of size {largest}"
            )
        self.plan = plan
        self.batch_size = batch_size
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        blocks = [list(block) for block in self.plan.blocks]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(blocks)
        batches: list[list[int]] = []
        current: list[int] = []
        for block in blocks:
            if current and len(current) + len(block) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(block)
        if current:
            batches.append(current)
        flattened = [index for batch in batches for index in batch]
        if sorted(flattened) != list(range(len(self.plan.sample_ids))):
            raise RuntimeError("paired sampler lost or duplicated a dataset record")
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


def _group_indices(
    batch: Mapping[str, Any], name: str, expected_size: int
) -> list[tuple[int, ...]]:
    metadata = batch.get("counterfactual_groups")
    if metadata is None:
        return []
    groups = metadata.get(name, [])
    result = []
    batch_size = int(batch["mixture"].size(0))
    for group in groups:
        indices = tuple(int(index) for index in group)
        if len(indices) != expected_size or len(indices) != len(set(indices)):
            raise ValueError(f"invalid {name} indices in paired batch")
        if any(index < 0 or index >= batch_size for index in indices):
            raise ValueError(f"out-of-range {name} index in paired batch")
        result.append(indices)
    return result


def _oracle_evidence_probability(
    batch: Mapping[str, Any], frames: int, dtype: torch.dtype
) -> torch.Tensor:
    """Frame-level oracle for the union, preserving anchor/answer overlap.

    Paper-profile QCES uses independent anchor/answer logits, but CEE compares
    the rendered evidence support rather than role identity.  Its temporal
    delta is therefore defined on the overlap-preserving evidence union; role
    identity remains supervised by the ordinary per-role BCE/Dice objectives.
    Legacy exclusive-softmax checkpoints use this same union-level CEE target.
    """

    union = (batch["anchor_mask"] + batch["answer_mask"]).clamp_max(1.0)
    return F.adaptive_max_pool1d(union[:, None], frames).squeeze(1).to(dtype)


def _mean(values: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else zero


def _relative_l1(error: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    numerator = error.abs().flatten(start_dim=1).mean(dim=1)
    denominator = reference.abs().flatten(start_dim=1).mean(dim=1).clamp_min(1e-5)
    return (numerator / denominator).mean()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten(start_dim=1)
    right = right.flatten(start_dim=1)
    return F.cosine_similarity(left, right, dim=-1, eps=1e-8).mean()


def _stable_rmse(error: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """RMSE with a finite zero-error gradient and an exact zero value."""

    return (error.square().mean() + epsilon).sqrt() - epsilon**0.5


def counterfactual_objectives(
    output: QCESOutput,
    batch: Mapping[str, Any],
    *,
    transition_margin: float = 0.25,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, int]]:
    """Return differentiable paired losses, diagnostics, and group counts."""

    if transition_margin < 0 or transition_margin > 1:
        raise ValueError("transition_margin must be in [0, 1]")
    zero = output.evidence.new_zeros(())
    for name, tensor in (
        ("semantic_condition", output.composition.semantic_condition),
        ("role_logits", output.composition.role_logits),
        ("evidence_probability", output.composition.evidence_probability),
        ("no_evidence_logit", output.composition.no_evidence_logit),
        ("predicted_evidence", output.evidence),
        ("oracle_evidence", batch["evidence"]),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite tensor in counterfactual objective: {name}")
    losses = {name: zero for name in COUNTERFACTUAL_LOSS_NAMES}
    metrics: Dict[str, torch.Tensor] = {}
    role_probability = output.composition.role_probabilities
    evidence_probability = output.composition.evidence_probability
    no_evidence_probability = output.composition.no_evidence_logit.sigmoid()
    oracle_evidence_probability = _oracle_evidence_probability(
        batch, evidence_probability.size(1), evidence_probability.dtype
    )

    surface_pairs = _group_indices(batch, "surface_pairs", 2)
    if surface_pairs:
        semantic_mse = []
        semantic_cosine = []
        role_mae = []
        no_evidence_mae = []
        evidence_relative_l1 = []
        for left, right in surface_pairs:
            if not torch.equal(batch["mixture"][left], batch["mixture"][right]):
                raise ValueError(
                    "surface-control pair does not contain the same aligned audio"
                )
            if not torch.equal(batch["evidence"][left], batch["evidence"][right]):
                raise ValueError(
                    "surface-control pair changes its oracle evidence target"
                )
            if not torch.equal(
                batch["anchor_mask"][left], batch["anchor_mask"][right]
            ) or not torch.equal(
                batch["answer_mask"][left], batch["answer_mask"][right]
            ):
                raise ValueError("surface-control pair changes its oracle roles")
            semantic_mse.append(
                F.mse_loss(
                    output.composition.semantic_condition[left],
                    output.composition.semantic_condition[right],
                )
            )
            semantic_cosine.append(
                F.cosine_similarity(
                    output.composition.semantic_condition[left][None],
                    output.composition.semantic_condition[right][None],
                    dim=-1,
                    eps=1e-8,
                ).mean()
            )
            role_mae.append(
                (role_probability[left] - role_probability[right]).abs().mean()
            )
            no_evidence_mae.append(
                (no_evidence_probability[left] - no_evidence_probability[right]).abs()
            )
            prediction_difference = (
                output.evidence[left] - output.evidence[right]
            )[None]
            oracle_scale = 0.5 * (
                batch["evidence"][left].abs() + batch["evidence"][right].abs()
            )[None]
            evidence_relative_l1.append(
                _relative_l1(prediction_difference, oracle_scale)
            )
        losses["surface_semantic_invariance"] = _mean(semantic_mse, zero)
        losses["surface_role_invariance"] = _mean(role_mae, zero)
        losses["surface_no_evidence_invariance"] = _mean(no_evidence_mae, zero)
        losses["surface_evidence_invariance"] = _mean(
            evidence_relative_l1, zero
        )
        metrics.update(
            surface_semantic_cosine=_mean(semantic_cosine, zero),
            surface_role_probability_mae=losses["surface_role_invariance"],
            surface_no_evidence_probability_mae=losses[
                "surface_no_evidence_invariance"
            ],
            surface_evidence_relative_l1=losses["surface_evidence_invariance"],
        )

    primary_triplets = _group_indices(batch, "primary_triplets", 3)
    if primary_triplets:
        temporal_errors = []
        temporal_cosines = []
        evidence_errors = []
        evidence_cosines = []
        transition_losses = []
        transition_accuracies = []
        for base, swap, drop in primary_triplets:
            for left, right in ((base, swap), (swap, drop), (base, drop)):
                predicted_temporal_delta = (
                    evidence_probability[right] - evidence_probability[left]
                )[None]
                oracle_temporal_delta = (
                    oracle_evidence_probability[right]
                    - oracle_evidence_probability[left]
                )[None]
                temporal_errors.append(
                    _stable_rmse(
                        predicted_temporal_delta - oracle_temporal_delta
                    )
                )
                temporal_cosines.append(
                    _cosine(predicted_temporal_delta, oracle_temporal_delta)
                )
                predicted_evidence_delta = (
                    output.evidence[right] - output.evidence[left]
                )[None]
                oracle_evidence_delta = (
                    batch["evidence"][right] - batch["evidence"][left]
                )[None]
                evidence_errors.append(
                    _relative_l1(
                        predicted_evidence_delta - oracle_evidence_delta,
                        oracle_evidence_delta,
                    )
                )
                evidence_cosines.append(
                    _cosine(predicted_evidence_delta, oracle_evidence_delta)
                )
            base_probability = no_evidence_probability[base]
            swap_probability = no_evidence_probability[swap]
            drop_probability = no_evidence_probability[drop]
            transition_losses.append(
                0.5
                * (
                    F.relu(transition_margin - (drop_probability - base_probability))
                    + F.relu(
                        transition_margin - (drop_probability - swap_probability)
                    )
                )
            )
            transition_accuracies.append(
                (
                    (base_probability < 0.5)
                    & (swap_probability < 0.5)
                    & (drop_probability >= 0.5)
                ).to(output.evidence.dtype)
            )
        losses["family_temporal_delta"] = _mean(temporal_errors, zero)
        losses["family_evidence_delta"] = _mean(evidence_errors, zero)
        losses["family_no_evidence_transition"] = _mean(
            transition_losses, zero
        )
        metrics.update(
            family_temporal_delta_rmse=losses["family_temporal_delta"],
            family_temporal_delta_cosine=_mean(temporal_cosines, zero),
            family_evidence_delta_relative_l1=losses["family_evidence_delta"],
            family_evidence_delta_cosine=_mean(evidence_cosines, zero),
            family_transition_accuracy=_mean(transition_accuracies, zero),
        )

    question_pairs = _group_indices(batch, "question_pairs", 2)
    if question_pairs:
        temporal_errors = []
        temporal_cosines = []
        evidence_errors = []
        evidence_cosines = []
        for left, right in question_pairs:
            if not torch.equal(batch["mixture"][left], batch["mixture"][right]):
                raise ValueError(
                    "same-scene question pair does not contain aligned audio"
                )
            predicted_temporal_delta = (
                evidence_probability[right] - evidence_probability[left]
            )[None]
            oracle_temporal_delta = (
                oracle_evidence_probability[right]
                - oracle_evidence_probability[left]
            )[None]
            oracle_evidence_delta = (
                batch["evidence"][right] - batch["evidence"][left]
            )[None]
            if bool((oracle_temporal_delta == 0).all()) and bool(
                (oracle_evidence_delta == 0).all()
            ):
                raise ValueError(
                    "same-scene question pair has a trivial oracle target delta"
                )
            temporal_errors.append(
                _stable_rmse(predicted_temporal_delta - oracle_temporal_delta)
            )
            temporal_cosines.append(
                _cosine(predicted_temporal_delta, oracle_temporal_delta)
            )
            predicted_evidence_delta = (
                output.evidence[right] - output.evidence[left]
            )[None]
            evidence_errors.append(
                _relative_l1(
                    predicted_evidence_delta - oracle_evidence_delta,
                    oracle_evidence_delta,
                )
            )
            evidence_cosines.append(
                _cosine(predicted_evidence_delta, oracle_evidence_delta)
            )
        losses["question_temporal_delta"] = _mean(temporal_errors, zero)
        losses["question_evidence_delta"] = _mean(evidence_errors, zero)
        metrics.update(
            question_temporal_delta_rmse=losses["question_temporal_delta"],
            question_temporal_delta_cosine=_mean(temporal_cosines, zero),
            question_evidence_delta_relative_l1=losses[
                "question_evidence_delta"
            ],
            question_evidence_delta_cosine=_mean(evidence_cosines, zero),
        )

    counts = {
        "surface": len(surface_pairs),
        "family": len(primary_triplets),
        "question": len(question_pairs),
    }
    return losses, metrics, counts
