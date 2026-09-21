import torch

from mixi_understanding.scripts.train_qces_overlap_aware_red_epn_v2 import (
    interval_hard_negative_rank_loss,
)


def test_interval_rank_masks_other_coactive_gold_labels() -> None:
    logits = torch.zeros(1, 4, 3)
    logits[:, :, 0] = 1.0
    logits[:, :, 1] = 2.0
    logits[:, :, 2] = 3.0
    presence = torch.zeros_like(logits)
    presence[:, :, 0] = 1.0
    presence[:, :, 2] = 1.0
    events = [[{"label_id": 0, "start": 0.0, "end": 1.0}]]
    loss = interval_hard_negative_rank_loss(
        logits, presence, events, margin=0.0, hard_negatives=2
    )
    # Class 2 is co-active and must not be the hard negative. Only class 1 is.
    assert torch.allclose(loss, torch.nn.functional.softplus(torch.tensor(1.0)))


def test_interval_rank_decreases_when_positive_logit_increases() -> None:
    presence = torch.zeros(1, 2, 2)
    presence[:, :, 0] = 1.0
    events = [[{"label_id": 0, "start": 0.0, "end": 1.0}]]
    low = interval_hard_negative_rank_loss(
        torch.zeros(1, 2, 2), presence, events, margin=0.5, hard_negatives=1
    )
    logits = torch.zeros(1, 2, 2)
    logits[:, :, 0] = 3.0
    high = interval_hard_negative_rank_loss(
        logits, presence, events, margin=0.5, hard_negatives=1
    )
    assert high < low
