from types import SimpleNamespace

import torch

from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
    center_width_to_intervals,
    hungarian_match_event_slots,
    interval_iou_matrix,
    relational_event_slots_v1_loss,
)
from mixi_understanding.scripts.train_qces_relational_event_slots_v1 import (
    execute_relational_query,
)


def test_interval_conversion_and_iou() -> None:
    interval = center_width_to_intervals(torch.tensor([[0.50, 0.20], [0.10, 0.40]]))
    assert torch.allclose(interval[0], torch.tensor([0.40, 0.60]))
    assert torch.allclose(interval[1], torch.tensor([0.00, 0.30]))
    iou = interval_iou_matrix(
        torch.tensor([[0.10, 0.30], [0.40, 0.60]]),
        torch.tensor([[0.20, 0.30], [0.40, 0.60]]),
    )
    assert torch.allclose(iou, torch.tensor([[0.5, 0.0], [0.0, 1.0]]))


def test_forward_matching_and_loss_are_finite() -> None:
    torch.manual_seed(7)
    model = RelationalEventSlotsV1(
        RelationalEventSlotsV1Config(
            feature_dim=12,
            hidden_dim=24,
            num_slots=4,
            num_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=48,
            dropout=0.0,
        )
    )
    features = torch.randn(2, 20, 12)
    valid = torch.ones(2, 20, dtype=torch.bool)
    outputs = model(features, valid)
    targets = [torch.tensor([[0.10, 0.20], [0.50, 0.65]]), torch.tensor([[0.30, 0.45]])]
    matches = hungarian_match_event_slots(
        outputs["objectness_logits"], outputs["intervals"], targets
    )
    assert [len(pair[0]) for pair in matches] == [2, 1]
    frame_event = torch.zeros(2, 20)
    frame_event[0, 2:4] = 1
    frame_event[0, 10:13] = 1
    frame_event[1, 6:9] = 1
    frame_onset = torch.zeros_like(frame_event)
    frame_onset[0, [2, 10]] = 1
    frame_onset[1, 6] = 1
    losses = relational_event_slots_v1_loss(
        outputs, targets, frame_event, frame_onset, valid
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.input_projection.weight.grad is not None


def test_executor_uses_anchor_only_and_respects_ordinal() -> None:
    slots = [
        {"slot_index": 0, "score": 0.9, "start": 0.10, "end": 0.20},
        {"slot_index": 1, "score": 0.9, "start": 0.35, "end": 0.45},
        {"slot_index": 2, "score": 0.9, "start": 0.60, "end": 0.70},
        {"slot_index": 3, "score": 0.9, "start": 0.80, "end": 0.90},
    ]
    logits = torch.full((250, 2), -5.0)
    logits[25:50, 0] = 8.0
    logits[150:175, 0] = 7.0
    # The second occurrence of class 0 is slot 2, so "after" points to slot 3.
    qa = SimpleNamespace(
        qa=SimpleNamespace(anchor_label="anchor", anchor_ordinal=2, relation="after")
    )
    result = execute_relational_query(slots, logits, qa, {"anchor": 0})
    assert result["anchor"]["slot_index"] == 2
    assert result["answer"]["slot_index"] == 3
    assert result["predicted_none"] is False

