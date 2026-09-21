from __future__ import annotations

import unittest

from mixi_understanding.data.qces_realdesed import (
    Event,
    build_question_records,
    choose_ten_second_crop,
    merge_reviewed_events,
    validate_inference_record,
    validate_scoring_record,
)


class RealDESEDEventContractTest(unittest.TestCase):
    def test_duplicate_and_touching_regions_merge_without_cross_class_merge(self) -> None:
        events = merge_reviewed_events(
            (
                Event(1.0, 2.0, "footsteps"),
                Event(1.0, 2.0, "footsteps"),
                Event(2.05, 2.5, "footsteps"),
                Event(1.2, 1.8, "running_water"),
            )
        )
        self.assertEqual(len(events), 2)
        by_label = {event.label: event for event in events}
        self.assertAlmostEqual(by_label["footsteps"].onset, 1.0)
        self.assertAlmostEqual(by_label["footsteps"].offset, 2.5)
        self.assertEqual({event.event_id for event in events}, {"event_000", "event_001"})

    def test_crop_is_event_rich_and_shifts_intervals_exactly_once(self) -> None:
        events = merge_reviewed_events(
            (
                Event(0.01, 0.2, "bell_ringing"),
                Event(11.0, 11.5, "footsteps"),
                Event(12.0, 12.5, "running_water"),
                Event(13.0, 13.5, "light_switch"),
            )
        )
        start, shifted = choose_ten_second_crop(
            events, source_duration_seconds=20.0
        )
        self.assertGreater(start, 0.0)
        self.assertEqual(
            {event.label for event in shifted},
            {"footsteps", "running_water", "light_switch"},
        )
        original_by_id = {event.event_id: event for event in events}
        for event in shifted:
            self.assertAlmostEqual(
                event.onset + start, original_by_id[event.event_id].onset, places=6
            )
            self.assertGreaterEqual(event.onset, 0.05 - 1e-6)
            self.assertLessEqual(event.offset, 9.95 + 1e-6)

    def test_long_event_is_right_censored_but_its_onset_remains_visible(self) -> None:
        events = merge_reviewed_events(
            (
                Event(2.0, 18.0, "vacuum_cleaner"),
                Event(4.0, 4.5, "phone_ringing"),
            )
        )
        start, shifted = choose_ten_second_crop(
            events, source_duration_seconds=20.0
        )
        by_label = {event.label: event for event in shifted}
        self.assertIn("vacuum_cleaner", by_label)
        self.assertAlmostEqual(
            by_label["vacuum_cleaner"].onset + start, 2.0, places=6
        )
        self.assertAlmostEqual(by_label["vacuum_cleaner"].offset, 9.95, places=6)

    def test_near_edge_event_with_no_visible_duration_is_excluded(self) -> None:
        events = merge_reviewed_events(
            (
                Event(1.0, 12.0, "running_water"),
                Event(2.0, 2.5, "footsteps"),
                Event(11.9499995, 12.2, "door_open_close"),
            )
        )
        _, shifted = choose_ten_second_crop(
            events, source_duration_seconds=20.0
        )
        for event in shifted:
            self.assertGreaterEqual(event.offset - event.onset, 0.05 - 1e-7)


class RealDESEDQuestionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = (
            Event(1.0, 1.5, "bell_ringing", "event_000"),
            Event(2.0, 3.0, "footsteps", "event_001"),
            Event(4.0, 5.5, "running_water", "event_002"),
        )

    def _records(self):
        return build_question_records(
            scene_id="realdesed_val_abc",
            upstream_record_id="example.wav",
            upstream_split="validation",
            mixture_path="audio/realdesed_val_abc.wav",
            mixture_sha256="a" * 64,
            source_license="CC BY 4.0",
            events=self.events,
        )

    def test_inference_is_label_free_and_scoring_is_bound(self) -> None:
        inference, scoring = self._records()
        self.assertEqual(len(inference), 8)
        self.assertEqual(len(scoring), 8)
        scoring_only = {
            "answer",
            "answer_options",
            "answer_event_ids",
            "evidence_intervals",
            "source_license",
        }
        for inference_row, scoring_row in zip(inference, scoring):
            validate_inference_record(inference_row)
            validate_scoring_record(scoring_row)
            self.assertTrue(scoring_only.isdisjoint(inference_row))

    def test_real_rows_forbid_oracle_stem_and_waveform_sdr_claims(self) -> None:
        _, scoring = self._records()
        invalid = dict(scoring[0])
        invalid["waveform_sdr_evaluation_allowed"] = True
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_scoring_record(invalid)

        invalid = dict(scoring[0])
        invalid["evidence_stem_path"] = "oracle.wav"
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_scoring_record(invalid)

    def test_binding_breaks_when_question_is_modified(self) -> None:
        _, scoring = self._records()
        invalid = dict(scoring[0])
        invalid["question"] = "A modified question?"
        with self.assertRaisesRegex(ValueError, "not bound"):
            validate_scoring_record(invalid)

    def test_repeated_class_is_not_used_as_ambiguous_anchor(self) -> None:
        events = self.events + (
            Event(7.0, 7.4, "footsteps", "event_003"),
        )
        inference, _ = build_question_records(
            scene_id="realdesed_val_repeat",
            upstream_record_id="repeat.wav",
            upstream_split="validation",
            mixture_path="audio/repeat.wav",
            mixture_sha256="b" * 64,
            source_license="CC0",
            events=events,
        )
        answerable = [row for row in inference if row["relation"] != "no_evidence"]
        self.assertTrue(answerable)
        self.assertTrue(all("footsteps" not in row["question"] for row in answerable))


if __name__ == "__main__":
    unittest.main()
