from mixi_understanding.scripts.train_qces_gold_natural_v3_listwise_ranker import (
    NUMERIC_DIM,
    candidate_features,
)


def test_candidate_features_orient_before_after_distance() -> None:
    anchor = {"start": 0.5, "end": 0.6, "score": 0.9, "label_id": 1}
    before = {"start": 0.2, "end": 0.3, "score": 0.8, "label_id": 2}
    after = {"start": 0.7, "end": 0.8, "score": 0.8, "label_id": 2}
    left = candidate_features(
        before, anchor, relation=0, global_rank=0, temporal_rank=0, max_candidates=120
    )
    right = candidate_features(
        after, anchor, relation=1, global_rank=0, temporal_rank=0, max_candidates=120
    )
    assert len(left) == len(right) == NUMERIC_DIM
    assert abs(left[7] - 0.3) < 1e-6
    assert abs(right[7] - 0.2) < 1e-6
