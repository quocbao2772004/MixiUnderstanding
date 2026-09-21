import torch

from mixi_understanding.scripts.train_qces_gold_natural_v3_acoustic_pair_ranker_v2 import (
    AcousticPairNoneRanker,
    AcousticRankerConfig,
    build_question_examples,
    metrics_from_outputs,
)


def _proposal(label: int, start: float, end: float, score: float = 0.8) -> dict:
    return {
        "label_id": label,
        "start": start,
        "end": end,
        "score": score,
        "clip_score": score,
    }


def test_relation_violating_proposal_is_kept_for_soft_reranking() -> None:
    scene = {
        "events": [
            {"label_id": 1, "start": 0.2, "end": 0.3},
            {"label_id": 2, "start": 0.4, "end": 0.5},
        ],
        "proposals": [
            _proposal(1, 0.45, 0.55),
            # Correct answer label/span, but its predicted onset precedes the
            # predicted anchor onset. V1 discarded this candidate entirely.
            _proposal(2, 0.4, 0.5),
        ],
        "proposal_valid": [False, True],
        "features": torch.zeros(250, 8),
    }
    config = AcousticRankerConfig(
        feature_dim=8,
        num_classes=3,
        max_proposals=4,
        max_pairs=4,
        max_anchors=1,
        span_hidden_dim=4,
        span_dim=4,
    )
    examples, _ = build_question_examples([scene], config, iou_threshold=0.30)
    after = examples[0]
    assert after["candidate_mask"].sum() == 2  # one pair + one anchor/NONE
    assert after["candidate_numeric"][0, 13] == 0  # relation_valid feature


def test_no_evidence_target_selects_a_localized_anchor_none() -> None:
    scene = {
        "events": [
            {"label_id": 1, "start": 0.2, "end": 0.3},
            {"label_id": 2, "start": 0.4, "end": 0.5},
        ],
        "proposals": [
            _proposal(1, 0.2, 0.3),
            _proposal(2, 0.4, 0.5),
        ],
        "proposal_valid": [True, True],
        "features": torch.zeros(250, 8),
    }
    config = AcousticRankerConfig(
        feature_dim=8,
        num_classes=3,
        max_proposals=4,
        max_pairs=4,
        max_anchors=1,
        span_hidden_dim=4,
        span_dim=4,
    )
    examples, _ = build_question_examples([scene], config, iou_threshold=0.30)
    no_evidence = examples[-1]
    selected = no_evidence["target"] > 0
    assert selected.sum() == 1
    assert no_evidence["candidate_is_none"][selected].all()
    assert torch.allclose(
        no_evidence["candidate_anchor_start"][selected], torch.tensor([0.4])
    )


def test_span_encoder_shapes_and_masks_padding() -> None:
    config = AcousticRankerConfig(
        feature_dim=8,
        num_classes=3,
        max_proposals=4,
        max_pairs=4,
        max_anchors=1,
        span_samples=4,
        span_hidden_dim=4,
        span_dim=5,
    )
    model = AcousticPairNoneRanker(config)
    features = torch.randn(2, 10, 8)
    starts = torch.tensor([[0.0, 0.5], [0.2, 0.0]])
    ends = torch.tensor([[0.4, 0.9], [0.7, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    representation, logits = model.encode_spans(features, starts, ends, mask)
    assert representation.shape == (2, 2, 5)
    assert logits.shape == (2, 2, 3)
    assert torch.count_nonzero(representation[1, 1]) == 0


def test_metrics_require_anchor_and_answer_for_joint_evidence() -> None:
    outputs = {
        "scores": torch.tensor([[2.0, 0.0]]),
        "candidate_is_none": torch.tensor([[False, True]]),
        "candidate_anchor_start": torch.tensor([[0.7, 0.2]]),
        "candidate_anchor_end": torch.tensor([[0.8, 0.3]]),
        "candidate_answer_start": torch.tensor([[0.4, 0.0]]),
        "candidate_answer_end": torch.tensor([[0.5, 0.0]]),
        "candidate_answer_label": torch.tensor([[2, -1]]),
        "no_evidence": torch.tensor([False]),
        "gold_anchor_start": torch.tensor([0.2]),
        "gold_anchor_end": torch.tensor([0.3]),
        "gold_answer_label": torch.tensor([2]),
        "gold_answer_start": torch.tensor([0.4]),
        "gold_answer_end": torch.tensor([0.5]),
    }
    metrics = metrics_from_outputs(outputs, none_bias=0.0, iou_threshold=0.30)
    assert metrics["answerable_answer_event_iou030_accuracy_↑"] == 1.0
    assert metrics["answerable_joint_evidence_iou030_accuracy_↑"] == 0.0
