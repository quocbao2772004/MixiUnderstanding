from mixi_understanding.scripts.audit_qces_overlap_aware_red_epn_v2 import (
    adjacent_index_pairs,
    event_hit,
)


def test_event_hit_requires_matching_label_and_iou() -> None:
    event = {"label_id": 2, "start": 0.1, "end": 0.3}
    proposals = [
        {"label_id": 1, "start": 0.1, "end": 0.3},
        {"label_id": 2, "start": 0.6, "end": 0.8},
        {"label_id": 2, "start": 0.12, "end": 0.31},
    ]
    assert not event_hit(proposals, event, top_k=2, iou_threshold=0.30)
    assert event_hit(proposals, event, top_k=3, iou_threshold=0.30)


def test_adjacent_index_pairs_skip_tied_onsets() -> None:
    events = [
        {"start": 0.0, "end": 0.5},
        {"start": 0.0, "end": 0.8},
        {"start": 0.4, "end": 0.9},
    ]
    assert adjacent_index_pairs(events) == [(1, 2)]
