import torch

from mixi_understanding.qces.overlap_event_slots_v2 import (
    OverlapEventSlotsV2,
    decode_overlap_event_slots_v2,
    overlap_event_slots_v2_loss,
)
from mixi_understanding.qces.relational_event_slots_v1 import RelationalEventSlotsV1Config


def test_overlap_slots_forward_and_loss() -> None:
    torch.manual_seed(17)
    model = OverlapEventSlotsV2(
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
    assert outputs["slot_mask_logits"].shape == (2, 4, 20)
    targets = [torch.tensor([[0.10, 0.45], [0.30, 0.65]]), torch.tensor([[0.20, 0.50]])]
    masks = [torch.zeros(2, 20), torch.zeros(1, 20)]
    masks[0][0, 2:9] = 1
    masks[0][1, 6:13] = 1
    masks[1][0, 4:10] = 1
    frame_event = torch.stack((masks[0].amax(0), masks[1].amax(0)))
    frame_onset = torch.zeros_like(frame_event)
    frame_onset[0, [2, 6]] = 1
    frame_onset[1, 4] = 1
    losses = overlap_event_slots_v2_loss(
        outputs, targets, masks, frame_event, frame_onset, valid
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.slot_mask_query[1].weight.grad is not None


def test_decoder_keeps_overlapping_slots() -> None:
    outputs = {
        "objectness_logits": torch.tensor([[8.0, 7.0, -8.0]]),
        "intervals": torch.tensor([[[0.10, 0.50], [0.30, 0.70], [0.80, 0.90]]]),
        "slot_mask_logits": torch.zeros(1, 3, 20),
    }
    decoded = decode_overlap_event_slots_v2(outputs, objectness_threshold=0.5)
    assert len(decoded[0]) == 2
    assert decoded[0][0]["end"] > decoded[0][1]["start"]
