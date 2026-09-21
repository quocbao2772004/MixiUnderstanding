import torch

from mixi_understanding.qces.class_aware_red_epn_v1 import (
    ClassAwareRedEpnV1,
    ClassAwareRedEpnV1Config,
    class_aware_red_epn_v1_loss,
    decode_class_aware_proposals,
    red_event_probabilities,
)


def test_red_recurrence_masks_invalid_frames_and_is_differentiable() -> None:
    start = torch.full((1, 6, 2), -4.0, requires_grad=True)
    end = torch.full((1, 6, 2), -4.0, requires_grad=True)
    valid = torch.tensor([[True, True, True, True, False, False]])
    outputs = red_event_probabilities(start, end, valid)
    presence = outputs["red_presence_probability"]
    assert presence.shape == (1, 6, 2)
    assert torch.all((presence >= 0.0) & (presence <= 1.0))
    assert torch.count_nonzero(presence[:, 4:]) == 0
    presence[:, :4].sum().backward()
    assert start.grad is not None
    assert end.grad is not None


def test_class_aware_model_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(11)
    config = ClassAwareRedEpnV1Config(
        feature_dim=12,
        num_classes=5,
        context_dim=16,
        epn_hidden_dim=8,
        context_kernel_size=3,
        dropout=0.0,
        minimum_duration=1.0 / 20.0,
    )
    model = ClassAwareRedEpnV1(config)
    features = torch.randn(2, 20, 12)
    detector_logits = torch.randn(2, 20, 5)
    valid = torch.ones(2, 20, dtype=torch.bool)
    valid[1, 17:] = False
    outputs = model(features, detector_logits, valid)

    presence = torch.zeros(2, 20, 5)
    onset = torch.zeros_like(presence)
    offset = torch.zeros_like(presence)
    since = torch.zeros_like(presence)
    until = torch.zeros_like(presence)
    center = ((torch.arange(20).float() + 0.5) / 20.0)[None, :, None].expand(2, -1, 5)
    clip = torch.zeros(2, 5)
    for batch, label, left, right in ((0, 1, 3, 9), (0, 3, 6, 13), (1, 2, 5, 12)):
        presence[batch, left:right, label] = 1.0
        onset[batch, left, label] = 1.0
        offset[batch, right - 1, label] = 1.0
        event_start = left / 20.0
        event_end = right / 20.0
        since[batch, left:right, label] = center[batch, left:right, label] - event_start
        until[batch, left:right, label] = event_end - center[batch, left:right, label]
        clip[batch, label] = 1.0
    targets = {
        "frame_presence": presence,
        "frame_onset": onset,
        "frame_offset": offset,
        "duration_since_onset": since,
        "duration_to_offset": until,
        "frame_center": center,
        "clip_presence": clip,
    }
    losses = class_aware_red_epn_v1_loss(outputs, targets, valid)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert model.context_projection.weight.grad is not None
    assert model.duration_head.weight.grad is not None


def test_decoder_keeps_simultaneous_events_with_different_labels() -> None:
    # Both classes deliberately occupy the same interval. Cross-class NMS
    # would delete one of them; class-aware decoding must retain both.
    presence = torch.full((1, 8, 2), 0.05)
    presence[0, 3, :] = 0.95
    start = torch.full((1, 8, 2), 0.25)
    end = torch.full((1, 8, 2), 0.75)
    outputs = {
        "fused_presence_probability": presence,
        "proposal_start": start,
        "proposal_end": end,
        "clip_presence_logits": torch.tensor([[5.0, 5.0]]),
    }
    proposals = decode_class_aware_proposals(
        outputs,
        0,
        max_classes=2,
        candidate_frames_per_class=1,
        max_events_per_class=1,
        max_events=2,
    )
    assert {int(row["label_id"]) for row in proposals} == {0, 1}
    assert all(float(row["start"]) == 0.25 for row in proposals)
    assert all(float(row["end"]) == 0.75 for row in proposals)
