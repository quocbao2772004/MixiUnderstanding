from pathlib import Path

import torch

from mixi_understanding.scripts.train_qces_beats_event_rank_v1 import (
    RankScene,
    event_hard_negative_loss,
    scene_hard_negative_loss,
    trust_region_losses,
)


def _scene(domain: str = "overlap") -> RankScene:
    return RankScene(
        scene_id="scene",
        domain=domain,
        mixture_path=Path("unused.wav"),
        valid_frames=10,
        events=(
            {"label_id": 0, "onset_seconds": 0.08, "offset_seconds": 0.28},
            {"label_id": 1, "onset_seconds": 0.12, "offset_seconds": 0.32},
        ),
    )


def test_event_rank_excludes_simultaneous_gold_and_rewards_target() -> None:
    logits = torch.zeros(1, 10, 4)
    baseline = event_hard_negative_loss(logits, [_scene()], margin=1.0, hard_negatives=2)
    improved = logits.clone()
    improved[:, :, 0:2] = 3.0
    improved_loss = event_hard_negative_loss(
        improved, [_scene()], margin=1.0, hard_negatives=2
    )
    assert improved_loss < baseline

    # The two overlapping gold classes must not be treated as negatives for
    # one another. Raising either by the same amount leaves their loss equal.
    swapped = improved.clone()
    swapped[:, :, 0] = 4.0
    swapped[:, :, 1] = 4.0
    swapped_loss = event_hard_negative_loss(
        swapped, [_scene()], margin=1.0, hard_negatives=2
    )
    assert swapped_loss < improved_loss


def test_scene_rank_rewards_all_present_labels() -> None:
    logits = torch.zeros(1, 10, 4)
    baseline = scene_hard_negative_loss(logits, [_scene()], margin=1.0, hard_negatives=2)
    logits[:, :, 0:2] = 2.0
    assert scene_hard_negative_loss(logits, [_scene()], margin=1.0, hard_negatives=2) < baseline


def test_trust_region_only_penalizes_absent_overshoot() -> None:
    teacher = torch.zeros(1, 10, 4)
    student = torch.zeros_like(teacher)
    target = torch.zeros_like(teacher)
    target[:, 2:7, 0] = 1.0
    valid = torch.ones(1, 10, dtype=torch.bool)
    student[:, 2:7, 0] = 2.0  # allowed positive improvement
    anti_positive, _ = trust_region_losses(
        student, teacher, target, valid, [_scene("nonoverlap")]
    )
    student[:, :, 3] = 2.0  # absent-class false-positive overshoot
    anti_false, retention = trust_region_losses(
        student, teacher, target, valid, [_scene("nonoverlap")]
    )
    assert anti_false > anti_positive
    assert retention > 0

    _, empty_retention = trust_region_losses(
        student, teacher, target, valid, [_scene("overlap")]
    )
    assert torch.isfinite(empty_retention)
    assert empty_retention == 0
