from dataclasses import replace
from pathlib import Path

import torch

from mixi_understanding.qces.clean_evidence_scenes import CleanSource
from mixi_understanding.scripts.build_qces_overlap_gold_natural_v3 import (
    HOP_SAMPLES,
    _highest_energy_window,
    natural_cluster_starts,
    select_eligible_labels,
)
from mixi_understanding.scripts import build_qces_overlap_query_train_dev_v1 as v1


def _source(source_id: str, label: str, tier: str = "gold") -> CleanSource:
    return CleanSource(
        source_id=source_id,
        label=label,
        official_split="train",
        video_id=source_id,
        source_sha256=source_id,
        audio_path=str(Path("unused.wav")),
        active_onset_seconds=0.0,
        active_offset_seconds=1.0,
        cleanliness_passed=True,
        audibility_passed=True,
        cleanliness_tier=tier,
        audibility_score=1.0,
        provenance={},
    )


def test_eligibility_uses_gold_source_counts_only() -> None:
    train = [_source(f"a{i}", "a") for i in range(8)] + [
        _source(f"b{i}", "b") for i in range(7)
    ]
    dev = [_source(f"ad{i}", "a") for i in range(3)] + [
        _source(f"bd{i}", "b") for i in range(3)
    ]
    train.append(_source("silver", "b", "silver"))
    labels, report = select_eligible_labels(
        ["a", "b"], train, dev, min_train=8, min_dev=3
    )
    assert labels == ["a"]
    assert report["excluded"][0]["gold_train_sources"] == 7


def test_highest_energy_window_is_deterministic_and_does_not_stretch() -> None:
    waveform = torch.zeros(4 * HOP_SAMPLES)
    waveform[2 * HOP_SAMPLES : 3 * HOP_SAMPLES] = 2.0
    crop, start = _highest_energy_window(waveform, HOP_SAMPLES)
    assert start == 2 * HOP_SAMPLES
    assert crop.numel() == HOP_SAMPLES
    assert torch.all(crop == 2.0)


def test_natural_cluster_starts_preserves_short_transient_concurrency() -> None:
    import random

    for lengths in ([1, 10, 10], [2, 8, 9], [1, 1], [3, 3, 3]):
        kind = "triple_overlap" if len(lengths) == 3 else "pair_overlap"
        starts = natural_cluster_starts(
            lengths, kind=kind, overlap=0.25, rng=random.Random(7)
        )
        events = [
            {
                "onset_seconds": start * 0.04,
                "offset_seconds": (start + length) * 0.04,
            }
            for start, length in zip(starts, lengths, strict=True)
        ]
        assert v1._maximum_concurrency(events) == len(lengths)
