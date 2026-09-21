from mixi_understanding.apps.qces_acoustic_ranker_mentor_demo import (
    normalize_query_text,
    parse_direct_question,
)


SCENE_ROWS = [
    {
        "anchor_label": "Microwave_oven",
        "anchor_display_label": "Microwave oven",
        "relation": "after",
        "predicted_answer_label": "Printer",
    },
    {
        "anchor_label": "Microwave_oven",
        "anchor_display_label": "Microwave oven",
        "relation": "before",
        "predicted_answer_label": "Slam",
    },
]


def test_normalize_query_text_handles_vietnamese_and_taxonomy_names() -> None:
    assert normalize_query_text("Âm gì SAU Microwave_oven?") == "am gi sau microwave oven"


def test_parse_direct_question_supports_english_and_vietnamese() -> None:
    prediction, parsed = parse_direct_question(
        "What sound occurs after Microwave oven?", SCENE_ROWS
    )
    assert prediction is SCENE_ROWS[0]
    assert parsed == {
        "reason": "ok",
        "relation": "after",
        "anchor": "Microwave_oven",
    }

    prediction, parsed = parse_direct_question(
        "Âm thanh nào xuất hiện trước Microwave oven?", SCENE_ROWS
    )
    assert prediction is SCENE_ROWS[1]
    assert parsed["relation"] == "before"


def test_parse_direct_question_rejects_unsupported_ordinal() -> None:
    prediction, parsed = parse_direct_question(
        "What happens after the second Microwave oven?", SCENE_ROWS
    )
    assert prediction is None
    assert parsed["reason"] == "ordinal_not_supported_by_checkpoint"
