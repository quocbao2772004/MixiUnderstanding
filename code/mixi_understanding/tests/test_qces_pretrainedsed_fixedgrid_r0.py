from __future__ import annotations

import math

import torch

from mixi_understanding.qces.fixed_grid_detector import (
    FRAME_HOP_SECONDS,
    NUM_FRAMES,
    build_fixed_grid_targets,
    masked_balanced_bce,
    valid_frames_for_duration,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector_fixedgrid_r0 import (
    decode_for_item,
)


def event(duration: float) -> dict[str, object]:
    return {
        "label": "event",
        "label_id": 0,
        "onset_seconds": 0.0,
        "offset_seconds": duration,
    }


def test_r0_duration_mapping_uses_fixed_40ms_grid() -> None:
    for duration in (0.5, 1.0, 2.227, 10.0):
        target, valid = build_fixed_grid_targets(
            [{"duration_seconds": duration, "events": [event(duration)]}], num_labels=1
        )
        expected = min(NUM_FRAMES, math.ceil(duration / FRAME_HOP_SECONDS - 1e-5))
        assert valid_frames_for_duration(duration) == expected
        assert int(valid.sum()) == expected
        assert int(target.sum()) == expected
        assert bool(target[0, :expected, 0].all())
        assert not bool(target[0, expected:, 0].any())


def test_r0_padding_has_zero_loss_gradient() -> None:
    target, valid = build_fixed_grid_targets(
        [{"duration_seconds": 1.0, "events": [event(1.0)]}], num_labels=1
    )
    logits = torch.zeros((1, NUM_FRAMES, 1), requires_grad=True)
    loss = masked_balanced_bce(
        logits, target, valid, pos_weight=torch.ones(1), class_weight=None
    )
    loss.backward()
    assert float(logits.grad[:, 25:].abs().sum()) == 0.0


def test_r0_decode_roundtrip_clips_to_real_duration() -> None:
    duration = 2.227
    valid_frames = valid_frames_for_duration(duration)
    probs = torch.zeros(NUM_FRAMES, 1)
    probs[:valid_frames, 0] = 1.0
    decoded = decode_for_item(
        probs,
        {
            "duration_seconds": duration,
            "valid_frames": valid_frames,
        },
        ["event"],
        threshold=0.5,
        min_duration=0.08,
        merge_gap=0.12,
    )
    assert len(decoded) == 1
    assert decoded[0]["onset_seconds"] == 0.0
    assert decoded[0]["offset_seconds"] == duration
