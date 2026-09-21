"""Source-free regression tests for the shortcut-resistant QCES v4 design."""

from __future__ import annotations

import unittest
from collections import Counter

from mixi_understanding.scripts.build_qces_v4_dataset import (
    NO_EVIDENCE_ANSWER,
    _answer_options,
    _label_blocks,
    _negative_labels,
    _question_specs,
)


class QCESV4BuilderDesignTest(unittest.TestCase):
    @staticmethod
    def _semantic_events() -> list[dict[str, str]]:
        return [
            {
                "event_id": f"event_semantic_{index}",
                "label": label,
                "event_kind": "semantic",
            }
            for index, label in enumerate(
                ("Croak", "Rain", "Printer", "Mechanical bell")
            )
        ]

    def test_question_set_balances_negative_operators(self) -> None:
        specs = _question_specs(
            self._semantic_events(),
            ("Chainsaw", "Fire alarm"),
            scene_number=0,
            seed=271828,
        )
        self.assertEqual(len(specs), 16)
        self.assertEqual(
            Counter(spec["question_type"] for spec in specs),
            Counter(
                {
                    "temporal_after": 3,
                    "temporal_before": 3,
                    "temporal_first": 6,
                    "no_evidence_after": 2,
                    "no_evidence_before": 2,
                }
            ),
        )
        first_specs = [spec for spec in specs if spec["relation"] == "first"]
        self.assertEqual(
            Counter(spec["query_labels"].index(spec["answer"]) for spec in first_specs),
            Counter({0: 3, 1: 3}),
        )

    def test_first_options_always_contain_both_named_candidates(self) -> None:
        spec = {
            "answer": "Croak",
            "relation": "first",
            "query_labels": ["Rain", "Croak"],
        }
        options = _answer_options(spec, "train_000000_00", 271828)
        self.assertEqual(len(options), 5)
        self.assertEqual(len(set(options)), 5)
        self.assertTrue(set(spec["query_labels"]).issubset(options))
        self.assertIn(NO_EVIDENCE_ANSWER, options)

    def test_no_evidence_options_are_not_duplicated(self) -> None:
        spec = {
            "answer": NO_EVIDENCE_ANSWER,
            "relation": "after",
            "query_labels": ["Steam whistle"],
        }
        options = _answer_options(spec, "train_000000_13", 271828)
        self.assertEqual(len(options), 5)
        self.assertEqual(len(set(options)), 5)
        self.assertEqual(options.count(NO_EVIDENCE_ANSWER), 1)

    def test_answer_option_positions_are_cyclically_balanced(self) -> None:
        spec = {
            "answer": "Croak",
            "relation": "after",
            "query_labels": ["Rain"],
        }
        positions = []
        for question_index in range(16):
            sample_id = f"test_000000_{question_index:02d}"
            options = _answer_options(spec, sample_id, 271828)
            positions.append(options.index(spec["answer"]))
        counts = Counter(positions)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_every_label_is_both_present_and_absent_queried(self) -> None:
        blocks = _label_blocks(271828)
        negative = _negative_labels(blocks, 271828)
        present = Counter(label for scenes in blocks.values() for scene in scenes for label in scene)
        absent = Counter(label for labels in negative.values() for label in labels)
        self.assertEqual(set(present), set(absent))
        self.assertTrue(all(count >= 1 for count in absent.values()))
        train_absent = Counter(
            label
            for (split, _), labels in negative.items()
            if split == "train"
            for label in labels
        )
        evaluation_absent = Counter(
            label
            for (split, _), labels in negative.items()
            if split in {"val", "test"}
            for label in labels
        )
        self.assertEqual(set(train_absent.values()), {1})
        self.assertEqual(set(evaluation_absent.values()), {1})


if __name__ == "__main__":
    unittest.main()
