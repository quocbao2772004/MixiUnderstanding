from mixi_understanding.scripts.export_qces_gold_natural_v3_acoustic_ranker_demo import (
    classify_case,
    display_label,
)


def test_display_label_is_human_readable() -> None:
    assert display_label("Keys_jangling") == "Keys jangling"
    assert display_label("Chirp_and_tweet") == "Chirp & tweet"


def test_case_classification_separates_joint_and_partial_success() -> None:
    assert (
        classify_case(
            no_evidence=False,
            predicted_none=False,
            anchor_correct=True,
            label_correct=True,
            answer_correct=True,
        )
        == "correct_joint_evidence"
    )
    assert (
        classify_case(
            no_evidence=False,
            predicted_none=False,
            anchor_correct=False,
            label_correct=True,
            answer_correct=True,
        )
        == "correct_answer_wrong_anchor"
    )


def test_case_classification_requires_verified_anchor_for_no_evidence() -> None:
    assert (
        classify_case(
            no_evidence=True,
            predicted_none=True,
            anchor_correct=True,
            label_correct=False,
            answer_correct=False,
        )
        == "correct_no_evidence"
    )
    assert (
        classify_case(
            no_evidence=True,
            predicted_none=True,
            anchor_correct=False,
            label_correct=False,
            answer_correct=False,
        )
        == "unverified_no_evidence"
    )
