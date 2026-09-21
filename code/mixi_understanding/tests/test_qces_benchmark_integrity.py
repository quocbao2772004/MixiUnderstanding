from __future__ import annotations

import unittest

from mixi_understanding.qces.benchmark_integrity import (
    audit_class_support,
    audit_qa_event_ambiguity,
    audit_qa_shortcuts,
    audit_split_overlaps,
    build_balanced_qa_items,
)


def _scene(scene_id: str, labels: tuple[str, ...] = ("Camera", "Slam", "Printer")) -> dict:
    events = []
    for index, label in enumerate(labels):
        events.append(
            {
                "event_id": f"{scene_id}:{index}",
                "event_kind": "semantic",
                "label": label,
                "onset_seconds": 1.0 + index * 2.0,
                "offset_seconds": 1.5 + index * 2.0,
                "source_id": f"source:{scene_id}:{index}",
            }
        )
    return {
        "scene_id": scene_id,
        "duration_seconds": 8.0,
        "events": events,
    }


def test_balanced_builder_removes_relation_answerability_shortcut() -> None:
    rows = [_scene(f"scene-{index:04d}") for index in range(200)]
    items = build_balanced_qa_items(
        rows,
        max_answerable_per_scene=4,
        max_no_evidence_per_scene=2,
        seed=7,
    )
    audit = audit_qa_shortcuts(items)

    assert audit["relation_target_counts"] == {
        "after": {"answerable": 400, "no_evidence": 200},
        "before": {"answerable": 400, "no_evidence": 200},
    }
    assert audit["text_only_majority_baselines"]["relation"]["balanced_accuracy"] == 0.5
    assert not audit["relation_shortcut"]
    assert audit["temporal_pair_audit"]["relation_invalid"] == 0


def test_builder_drops_simultaneous_onset_pairs() -> None:
    row = _scene("tied", ("Horse", "Neigh", "Speech"))
    row["events"][1]["onset_seconds"] = row["events"][0]["onset_seconds"]
    row["events"][1]["offset_seconds"] = 2.0
    items = build_balanced_qa_items(
        [row],
        max_answerable_per_scene=8,
        max_no_evidence_per_scene=0,
    )
    audit = audit_qa_shortcuts(items)

    # The tied pair contaminates both event identities, so neither may become
    # an anchor for a headline evidence question.
    assert len(items) == 0
    assert audit["temporal_pair_audit"]["onset_ties"] == 0
    assert audit["temporal_pair_audit"]["relation_invalid"] == 0


def test_builder_excludes_overlapping_and_same_label_pairs_from_headline() -> None:
    row = _scene("ambiguous", ("Camera", "Camera", "Printer", "Slam"))
    row["events"][1]["onset_seconds"] = 1.6
    row["events"][1]["offset_seconds"] = 2.2
    row["events"][2]["onset_seconds"] = 2.0
    row["events"][2]["offset_seconds"] = 2.8
    row["events"][3]["onset_seconds"] = 3.0
    row["events"][3]["offset_seconds"] = 3.5

    items = build_balanced_qa_items(
        [row],
        max_answerable_per_scene=20,
        max_no_evidence_per_scene=0,
    )

    # Printer overlaps the second Camera, so even its otherwise disjoint pair
    # with Slam is acoustically non-unique and must be excluded.
    assert items == []
    assert audit_qa_shortcuts(items)["temporal_pair_audit"]["overlapping_anchor_answer"] == 0


def test_ambiguity_audit_verifies_balanced_builder_policy() -> None:
    row = _scene("ambiguous-audit", ("Camera", "Printer", "Slam"))
    row["events"][1]["onset_seconds"] = 1.25
    row["events"][1]["offset_seconds"] = 2.25

    safe_items = build_balanced_qa_items(
        [row],
        max_answerable_per_scene=8,
        max_no_evidence_per_scene=2,
        require_unambiguous_events=True,
    )
    unsafe_items = build_balanced_qa_items(
        [row],
        max_answerable_per_scene=8,
        max_no_evidence_per_scene=2,
        require_unambiguous_events=False,
    )

    safe_audit = audit_qa_event_ambiguity(safe_items, [row])
    unsafe_audit = audit_qa_event_ambiguity(unsafe_items, [row])
    assert safe_audit["passes"]
    assert safe_audit["ambiguous_items"] == 0
    assert not unsafe_audit["passes"]
    assert unsafe_audit["ambiguous_items"] > 0


def test_cap_one_does_not_always_select_the_same_relation() -> None:
    rows = [_scene(f"scene-{index:04d}") for index in range(200)]
    items = build_balanced_qa_items(
        rows,
        max_answerable_per_scene=1,
        max_no_evidence_per_scene=1,
        seed=11,
    )
    answerable_relations = {
        item.relation for item in items if not item.no_evidence
    }
    no_evidence_relations = {
        item.relation for item in items if item.no_evidence
    }
    assert answerable_relations == {"after", "before"}
    assert no_evidence_relations == {"after", "before"}


def test_no_evidence_has_anchor_and_verification_evidence() -> None:
    items = build_balanced_qa_items(
        [_scene("one")],
        max_answerable_per_scene=0,
        max_no_evidence_per_scene=2,
    )
    by_relation = {item.relation: item for item in items}
    assert by_relation["before"].gold_verification_interval == (0.0, 1.0)
    assert by_relation["before"].gold_evidence_intervals == ((0.0, 1.5),)
    assert by_relation["after"].gold_verification_interval == (5.5, 8.0)
    assert by_relation["after"].gold_evidence_intervals == ((5.0, 8.0),)


def test_overlap_audit_uses_audio_and_source_identity_not_only_scene_id() -> None:
    train = _scene("train-name", ("Camera", "Slam"))
    val = _scene("renamed-val", ("Camera", "Slam"))
    train["audio_sha256"] = "same-audio"
    val["audio_sha256"] = "same-audio"
    train["video_id"] = "same-video"
    val["video_id"] = "same-video"
    train["events"][0]["source_sha256"] = "same-source"
    val["events"][0]["source_sha256"] = "same-source"

    report = audit_split_overlaps({"train": [train], "val": [val]})

    assert not report["passes"]
    fields = report["pairs"]["train__val"]["fields"]
    assert fields["audio_sha256"]["count"] == 1
    assert fields["video_id"]["count"] == 1
    assert fields["event.source_sha256"]["count"] == 1


def test_class_support_is_counted_by_independent_scene() -> None:
    rows = []
    for index in range(3):
        row = _scene(f"scene-{index}", ("Camera", "Camera"))
        rows.append(row)
    report = audit_class_support(
        rows,
        ontology=("Camera", "Printer"),
        minimum_scenes=3,
        minimum_active_seconds=1.0,
    )
    by_label = {row["label"]: row for row in report["per_label"]}

    assert by_label["Camera"]["scenes"] == 3
    assert by_label["Camera"]["occurrences"] == 6
    assert by_label["Camera"]["ready"]
    assert by_label["Printer"]["scenes"] == 0
    assert not by_label["Printer"]["ready"]


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Expose the lightweight function tests to stdlib unittest as well."""

    del loader, tests, pattern
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
