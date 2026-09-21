"""Unit tests for the QCES-v6 parser, proposal decoding and relational planner.

These cover the parts whose failure would be silent: a parser that reads the
wrong ordinal, a decoder that merges two occurrences into one, and a planner
that picks the wrong neighbour.  Each of those produces a plausible-looking
evidence stem with a wrong answer, which no waveform metric alone would flag.
"""

from __future__ import annotations

import torch

from mixi_understanding.qces.event_proposals import (
    EventProposal,
    FEATURE_GROUPS,
    HEAD_FEATURE_DIM,
    ProposalHead,
    decode_proposals,
    head_features,
    zero_feature_groups,
)
from mixi_understanding.qces.question_parsing import (
    RELATION_AFTER,
    RELATION_BEFORE,
    RELATION_FIRST,
    parse_question,
)
from mixi_understanding.qces.relational_planner import plan_from_proposals
from mixi_understanding.qces.stem_features import (
    NUM_FRAMES,
    STEM_FEATURE_DIM,
    FrameGrid,
    intervals_to_frame_targets,
    waveform_frame_features,
)

TAXONOMY = (
    "Bark",
    "Breathing",
    "Camera",
    "Keys_jangling",
    "Coin_(dropping)",
    "Chirp_and_tweet",
)
GRID = FrameGrid(sample_rate=32_000)


def test_parser_reads_after_with_ordinal() -> None:
    parsed = parse_question(
        "The third Camera begins; what begins next?",
        ["no_evidence", "Bark", "Keys_jangling"],
        TAXONOMY,
    )
    assert parsed.ok
    assert parsed.relation == RELATION_AFTER
    assert parsed.anchor_label == "Camera"
    assert parsed.anchor_ordinal == 3


def test_parser_reads_before_without_confusing_the_ordinal_word() -> None:
    parsed = parse_question(
        "Before the first instance of Bark starts, which event starts last?",
        ["no_evidence", "Camera"],
        TAXONOMY,
    )
    assert parsed.relation == RELATION_BEFORE
    assert parsed.anchor_label == "Bark"
    assert parsed.anchor_ordinal == 1


def test_parser_treats_two_named_classes_as_the_first_relation() -> None:
    parsed = parse_question(
        "In start-time order, which comes first, Coin_(dropping) or Bark?",
        ["no_evidence", "Coin_(dropping)", "Bark"],
        TAXONOMY,
    )
    assert parsed.relation == RELATION_FIRST
    assert set(parsed.candidate_labels) == {"Coin_(dropping)", "Bark"}


def test_parser_does_not_match_a_label_inside_a_longer_token() -> None:
    parsed = parse_question(
        "What starts next once the first Chirp_and_tweet has started?",
        ["no_evidence", "Bark"],
        TAXONOMY,
    )
    assert parsed.anchor_label == "Chirp_and_tweet"


def test_query_labels_union_covers_anchor_and_options() -> None:
    parsed = parse_question(
        "What sound begins next after the first occurrence of Camera?",
        ["no_evidence", "Bark", "Breathing"],
        TAXONOMY,
    )
    assert set(parsed.query_labels) == {"Camera", "Bark", "Breathing"}


def _proposal(label: str, onset: float, offset: float) -> EventProposal:
    return EventProposal(label, onset, offset, 1.0)


def test_planner_after_selects_the_nearest_later_onset() -> None:
    parsed = parse_question(
        "What sound begins next after the first occurrence of Camera?",
        ["no_evidence", "Bark", "Breathing"],
        TAXONOMY,
    )
    plan = plan_from_proposals(
        parsed,
        [
            _proposal("Camera", 1.0, 1.9),
            _proposal("Bark", 2.0, 2.8),
            _proposal("Breathing", 4.0, 4.9),
        ],
    )
    assert not plan.no_evidence
    assert plan.answer_label == "Bark"
    assert plan.spans == ((1.0, 1.9), (2.0, 2.8))


def test_planner_counts_occurrences_for_the_ordinal() -> None:
    parsed = parse_question(
        "The third Camera begins; what begins next?",
        ["no_evidence", "Bark"],
        TAXONOMY,
    )
    plan = plan_from_proposals(
        parsed,
        [
            _proposal("Camera", 0.5, 1.0),
            _proposal("Camera", 1.5, 2.0),
            _proposal("Camera", 2.5, 3.0),
            _proposal("Bark", 3.5, 4.0),
        ],
    )
    assert plan.answer_label == "Bark"
    assert plan.spans == ((2.5, 3.0), (3.5, 4.0))


def test_planner_neighbour_may_repeat_the_anchor_label() -> None:
    """The benchmark orders adjacent events by onset, not by distinct class."""

    parsed = parse_question(
        "What sound begins next after the first occurrence of Camera?",
        ["no_evidence", "Camera", "Bark"],
        TAXONOMY,
    )
    plan = plan_from_proposals(
        parsed,
        [
            _proposal("Camera", 1.0, 1.8),
            _proposal("Camera", 2.0, 2.8),
            _proposal("Bark", 5.0, 5.8),
        ],
    )
    assert plan.answer_label == "Camera"


def test_planner_abstains_when_the_ordinal_occurrence_is_missing() -> None:
    parsed = parse_question(
        "The third Camera begins; what begins next?",
        ["no_evidence", "Bark"],
        TAXONOMY,
    )
    plan = plan_from_proposals(
        parsed, [_proposal("Camera", 0.5, 1.0), _proposal("Bark", 3.5, 4.0)]
    )
    assert plan.no_evidence
    assert plan.answer_label is None


def test_planner_first_needs_both_candidates_present() -> None:
    parsed = parse_question(
        "Which begins sooner in the clip, Bark or Breathing?",
        ["no_evidence", "Bark", "Breathing"],
        TAXONOMY,
    )
    present = plan_from_proposals(
        parsed, [_proposal("Breathing", 1.0, 1.5), _proposal("Bark", 3.0, 3.5)]
    )
    assert present.answer_label == "Breathing"
    assert present.spans == ((1.0, 1.5), (3.0, 3.5))
    absent = plan_from_proposals(parsed, [_proposal("Bark", 3.0, 3.5)])
    assert absent.no_evidence


def test_decoder_merges_a_short_gap_but_keeps_distinct_regions() -> None:
    activity = torch.zeros(1, NUM_FRAMES)
    activity[0, 50:80] = 1.0
    activity[0, 82:110] = 1.0  # 2-frame gap, below the merge threshold
    activity[0, 300:340] = 1.0
    proposals = decode_proposals(["Bark"], activity, GRID, threshold=0.5)
    assert len(proposals) == 2
    assert proposals[0].onset_seconds == GRID.frame_to_seconds(50)
    assert proposals[1].onset_seconds == GRID.frame_to_seconds(300)


def test_onset_split_recovers_two_adjacent_occurrences() -> None:
    activity = torch.zeros(1, NUM_FRAMES)
    activity[0, 100:200] = 1.0
    onsets = torch.zeros(1, NUM_FRAMES)
    onsets[0, 102] = 1.0
    onsets[0, 150] = 1.0
    merged = decode_proposals(["Camera"], activity, GRID, threshold=0.5)
    assert len(merged) == 1
    split = decode_proposals(
        ["Camera"], activity, GRID, threshold=0.5, onset_activity=onsets
    )
    assert len(split) == 2
    assert split[0].offset_seconds == split[1].onset_seconds


def test_decoder_drops_regions_below_the_minimum_duration() -> None:
    activity = torch.zeros(1, NUM_FRAMES)
    activity[0, 10:12] = 1.0
    assert decode_proposals(["Bark"], activity, GRID, threshold=0.5) == []


def test_head_features_are_permutation_equivariant() -> None:
    torch.manual_seed(0)
    stems = torch.rand(3, STEM_FEATURE_DIM, NUM_FRAMES)
    mixture = torch.rand(STEM_FEATURE_DIM, NUM_FRAMES)
    clap_mixture = torch.rand(3, NUM_FRAMES)
    clap_stem = torch.rand(3, 3)
    order = [2, 0, 1]
    direct = head_features(stems, mixture, clap_mixture, clap_stem)
    permuted = head_features(
        stems[order],
        mixture,
        clap_mixture[order],
        clap_stem[order][:, order],
    )
    assert torch.allclose(direct[order], permuted, atol=1e-5)


def test_head_features_have_the_declared_width() -> None:
    stems = torch.rand(2, STEM_FEATURE_DIM, NUM_FRAMES)
    mixture = torch.rand(STEM_FEATURE_DIM, NUM_FRAMES)
    features = head_features(stems, mixture)
    assert features.shape == (2, HEAD_FEATURE_DIM, NUM_FRAMES)
    assert torch.isfinite(features).all()


def test_zeroing_a_feature_group_touches_only_that_group() -> None:
    features = torch.ones(2, HEAD_FEATURE_DIM, NUM_FRAMES)
    masked = zero_feature_groups(features, ["clap"])
    for channel in FEATURE_GROUPS["clap"]:
        assert float(masked[:, channel].abs().sum()) == 0.0
    untouched = [
        channel
        for channel in range(HEAD_FEATURE_DIM)
        if channel not in FEATURE_GROUPS["clap"]
    ]
    assert float(masked[:, untouched].min()) == 1.0


def test_head_emits_three_aligned_outputs() -> None:
    torch.manual_seed(0)
    head = ProposalHead().eval()
    features = torch.rand(4, HEAD_FEATURE_DIM, NUM_FRAMES)
    with torch.inference_mode():
        frames, onsets, presence = head(features)
    assert frames.shape == (4, NUM_FRAMES)
    assert onsets.shape == (4, NUM_FRAMES)
    assert presence.shape == (4,)


def test_frame_features_track_a_burst() -> None:
    waveform = torch.zeros(320_000)
    waveform[96_000:128_000] = torch.randn(32_000) * 0.1
    features = waveform_frame_features(waveform)
    assert features.shape == (STEM_FEATURE_DIM, NUM_FRAMES)
    energy = features[0]
    assert float(energy[160:190].mean()) > 100 * float(energy[10:40].mean() + 1e-12)


def test_interval_rasterisation_matches_the_grid() -> None:
    target = intervals_to_frame_targets([(1.0, 2.0)], GRID)
    assert float(target.sum()) == 50.0
    assert float(target[50]) == 1.0
    assert float(target[49]) == 0.0
