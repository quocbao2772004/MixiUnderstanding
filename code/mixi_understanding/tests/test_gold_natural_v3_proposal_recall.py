from mixi_understanding.scripts.audit_qces_gold_natural_v3_proposal_recall import (
    adjacent_pairs,
    cap_events_per_label,
    proposal_rank,
)
from mixi_understanding.scripts.evaluate_qces_detector_pair_retrieval_qa import Segment


def _segment(label: str, onset: float, offset: float) -> Segment:
    return Segment(label, 0, onset, offset, 0.9)


def test_proposal_rank_requires_label_and_temporal_overlap() -> None:
    pool = [
        _segment("wrong", 1.0, 2.0),
        _segment("target", 5.0, 6.0),
        _segment("target", 1.1, 2.1),
    ]
    label_rank, segment_rank, best_iou = proposal_rank(
        pool, label="target", interval=(1.0, 2.0), iou_threshold=0.30
    )
    assert label_rank == 2
    assert segment_rank == 3
    assert best_iou > 0.80


def test_adjacent_pairs_excludes_tied_onsets() -> None:
    events = [
        {"event_id": "a", "onset_seconds": 0.0, "offset_seconds": 1.0},
        {"event_id": "b", "onset_seconds": 0.0, "offset_seconds": 0.5},
        {"event_id": "c", "onset_seconds": 2.0, "offset_seconds": 3.0},
    ]
    assert adjacent_pairs(events) == [(0, 2)]


def test_cap_events_per_label_preserves_global_score_order() -> None:
    pool = [
        _segment("a", 0.0, 1.0),
        _segment("a", 1.0, 2.0),
        _segment("b", 2.0, 3.0),
        _segment("a", 3.0, 4.0),
    ]
    assert cap_events_per_label(pool, 2) == pool[:3]
