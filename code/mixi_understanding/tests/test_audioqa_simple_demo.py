from mixi_understanding.apps.qces_audioqa_simple_demo import (
    answer_is_correct,
    oracle_inventory,
)
from mixi_understanding.audioqa_event_graph import answer_event_graph_question


EVENTS = [
    {"label": "Camera", "display_label": "Camera", "start_seconds": 0.2, "end_seconds": 0.8},
    {"label": "Slam", "display_label": "Slam", "start_seconds": 1.0, "end_seconds": 1.5},
]
TAXONOMY = ["Camera", "Slam"]


def test_oracle_inventory_and_answer_judging() -> None:
    inventory = oracle_inventory(EVENTS)
    predicted = answer_event_graph_question("What sound occurs first?", inventory, TAXONOMY)
    oracle = answer_event_graph_question("What sound occurs first?", inventory, TAXONOMY)
    assert answer_is_correct(predicted, oracle)


def test_list_answer_ignores_display_order() -> None:
    oracle = answer_event_graph_question(
        "What sounds are present?", oracle_inventory(EVENTS), TAXONOMY
    )
    reversed_inventory = list(reversed(oracle_inventory(EVENTS)))
    predicted = answer_event_graph_question(
        "What sounds are present?", reversed_inventory, TAXONOMY
    )
    assert answer_is_correct(predicted, oracle)
