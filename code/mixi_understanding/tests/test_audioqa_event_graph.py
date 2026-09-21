from mixi_understanding.audioqa_event_graph import answer_event_graph_question


INVENTORY = [
    {
        "label": "Camera",
        "display_label": "Camera",
        "score": 0.9,
        "occurrences": [
            {"start_seconds": 0.2, "end_seconds": 0.8},
            {"start_seconds": 2.0, "end_seconds": 2.4},
        ],
    },
    {
        "label": "Slam",
        "display_label": "Slam",
        "score": 0.8,
        "occurrences": [{"start_seconds": 1.0, "end_seconds": 1.4}],
    },
    {
        "label": "Printer",
        "display_label": "Printer",
        "score": 0.7,
        "occurrences": [{"start_seconds": 2.2, "end_seconds": 3.0}],
    },
]
TAXONOMY = ["Camera", "Slam", "Printer", "Meow"]


def test_list_exists_count_and_locate() -> None:
    assert answer_event_graph_question("What sounds are present?", INVENTORY, TAXONOMY).answer == "Camera, Slam, Printer"
    assert answer_event_graph_question("Is there a Meow?", INVENTORY, TAXONOMY).answer == "NO"
    assert answer_event_graph_question("How many Camera sounds?", INVENTORY, TAXONOMY).answer == "2"
    assert answer_event_graph_question("Camera ở đoạn nào?", INVENTORY, TAXONOMY).answer == "0.20–0.80s, 2.00–2.40s"


def test_temporal_and_overlap_queries() -> None:
    assert answer_event_graph_question("What is after Slam?", INVENTORY, TAXONOMY).answer == "Camera"
    assert answer_event_graph_question("What is between Slam and Printer?", INVENTORY, TAXONOMY).answer == "Camera"
    assert answer_event_graph_question("What overlaps Printer?", INVENTORY, TAXONOMY).answer == "Camera"


def test_first_last_and_longest() -> None:
    assert answer_event_graph_question("What sound is first?", INVENTORY, TAXONOMY).answer == "Camera"
    assert answer_event_graph_question("What sound is last?", INVENTORY, TAXONOMY).answer == "Printer"
    at_end = answer_event_graph_question(
        "What sounds at the end of audio?", INVENTORY, TAXONOMY
    )
    assert at_end.intent == "last"
    assert at_end.answer == "Printer"
    assert len(at_end.evidence) == 1
    begins = answer_event_graph_question(
        "What sounds begin in this audio?", INVENTORY, TAXONOMY
    )
    assert begins.intent == "first"
    assert begins.answer == "Camera"
    assert len(begins.evidence) == 1
    assert answer_event_graph_question("What sound is longest?", INVENTORY, TAXONOMY).answer == "Printer"


def test_unique_parenthetical_alias_resolves_anchor() -> None:
    inventory = [
        {
            "label": "Frying_(food)",
            "display_label": "Frying (food)",
            "score": 0.9,
            "occurrences": [{"start_seconds": 0.2, "end_seconds": 1.0}],
        },
        {
            "label": "Slam",
            "display_label": "Slam",
            "score": 0.8,
            "occurrences": [{"start_seconds": 1.2, "end_seconds": 1.6}],
        },
    ]
    result = answer_event_graph_question(
        "What sounds after frying?", inventory, ["Frying_(food)", "Slam"]
    )
    assert result.supported
    assert result.intent == "after"
    assert result.answer == "Slam"


def test_unrecognized_what_sounds_does_not_silently_fall_back_to_list() -> None:
    result = answer_event_graph_question(
        "What sounds mysteriously transform here?", INVENTORY, TAXONOMY
    )
    assert not result.supported
    assert result.intent == "unknown"
