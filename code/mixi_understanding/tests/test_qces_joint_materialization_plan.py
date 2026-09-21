from __future__ import annotations

import unittest

from mixi_understanding.qces.joint_materialization_plan import (
    JointPlanError,
    build_joint_materialization_plan,
)
from mixi_understanding.qces.supported_ontology import StrongEvent


def _event(video: str, label: str, mid: str, onset: float) -> StrongEvent:
    return StrongEvent(
        segment_id=f"{video}_0",
        video_id=video,
        mid=mid,
        label=label,
        display_name=label,
        onset_seconds=onset,
        offset_seconds=onset + 0.4,
    )


def _crop(
    split: str,
    video: str,
    label: str,
    mid: str,
    onset: float,
    *,
    lock: str,
) -> dict[str, object]:
    return {
        "metadata_split": split,
        "video_id": video,
        "segment_id": f"{video}_0",
        "source_event_uid": f"uid-{video}-{label}",
        "coverage_mid": mid,
        "coverage_label": label,
        "coverage_rank": 1,
        "ambiguity_tier": 0,
        "ambiguity_tier_name": "fully_isolated_all_strong_labels",
        "fully_isolated": True,
        "clean_source_eligible": True,
        "any_different_overlap_fraction": 0.0,
        "selected_distractor_overlap_fraction": 0.0,
        "event_onset_seconds": onset,
        "event_offset_seconds": onset + 0.4,
        "crop_start_seconds": max(0.0, onset - 0.1),
        "crop_end_seconds": onset + 0.5,
        "materialized": False,
        "materialized_audio_exists": False,
        "materialized_audio_path": "",
        "split_lock": lock,
        "strong_annotations": [],
        "selected_ontology_annotations": [],
        "supervision_policy": "test",
        "selection_key": f"key-{split}-{video}-{label}",
    }


class JointMaterializationPlanTests(unittest.TestCase):
    def _fixtures(self):
        rows = {
            "train": [
                _crop("train", "t1", "A", "/a", 0.1, lock="train"),
                _crop("train", "t1", "B", "/b", 0.8, lock="train"),
                _crop("train", "t2", "A", "/a", 0.2, lock="unassigned_train_pool"),
                _crop("train", "t3", "B", "/b", 0.3, lock="dev"),
            ],
            "eval": [
                _crop("eval", "e1", "A", "/a", 0.1, lock="test"),
                _crop("eval", "e1", "B", "/b", 0.8, lock="test"),
            ],
        }
        events = {
            "train": {
                "t1": [
                    _event("t1", "A", "/a", 0.1),
                    _event("t1", "Unselected", "/u", 0.45),
                    _event("t1", "B", "/b", 0.8),
                ],
                "t2": [_event("t2", "A", "/a", 0.2)],
                "t3": [_event("t3", "B", "/b", 0.3)],
            },
            "eval": {
                "e1": [
                    _event("e1", "A", "/a", 0.1),
                    _event("e1", "B", "/b", 0.8),
                ]
            },
        }
        return rows, events

    def test_merges_class_rows_and_preserves_full_events_and_locks(self) -> None:
        rows, events = self._fixtures()
        plans, receipt = build_joint_materialization_plan(
            rows_by_split=rows,
            events_by_split=events,
            selected_labels={"A", "B"},
            selected_mids={"/a", "/b"},
            target_train_per_class=2,
            target_eval_per_class=1,
            raw_plan_ids={"train": {"t1", "raw-only"}, "eval": {"e1"}},
        )
        self.assertEqual(len(plans["train"]), 3)
        self.assertEqual(len(plans["eval"]), 1)
        t1 = next(row for row in plans["train"] if row["video_id"] == "t1")
        self.assertEqual(t1["covers_deficit_labels"], ["A", "B"])
        self.assertEqual(t1["coverage_request_count"], 2)
        self.assertEqual(len(t1["crop_requests"]), 2)
        self.assertEqual(len(t1["events"]), 3)
        self.assertEqual(t1["labels"], ["A", "B"])
        self.assertEqual(t1["all_strong_labels"], ["A", "B", "Unselected"])
        self.assertFalse(t1["events"][1]["selected_ontology_label"])
        self.assertEqual(t1["split_lock"], "train")
        self.assertEqual(receipt["total_unique_videos"], 4)
        self.assertEqual(receipt["old_raw_plan_comparison"]["train"]["raw_only_skipped"], 1)
        self.assertEqual(receipt["train_eval_source_overlap"], 0)
        self.assertTrue(receipt["invariants"]["exact_200_class_100_20_contract"])

    def test_rejects_wrong_quota_and_cross_split_source_overlap(self) -> None:
        rows, events = self._fixtures()
        with self.assertRaisesRegex(JointPlanError, "exact class quotas"):
            build_joint_materialization_plan(
                rows_by_split=rows,
                events_by_split=events,
                selected_labels={"A", "B"},
                selected_mids={"/a", "/b"},
                target_train_per_class=3,
                target_eval_per_class=1,
            )
        rows["eval"] = [
            _crop("eval", "t1", "A", "/a", 0.1, lock="test"),
            _crop("eval", "t1", "B", "/b", 0.8, lock="test"),
        ]
        events["eval"] = {"t1": events["train"]["t1"]}
        with self.assertRaisesRegex(JointPlanError, "source overlap"):
            build_joint_materialization_plan(
                rows_by_split=rows,
                events_by_split=events,
                selected_labels={"A", "B"},
                selected_mids={"/a", "/b"},
                target_train_per_class=2,
                target_eval_per_class=1,
            )

    def test_rejects_coverage_label_mid_pair_mismatch(self) -> None:
        rows, events = self._fixtures()
        # Both the label and MID exist in the selected sets, but this swapped
        # pair is not a valid ontology/event identity.
        rows["train"][0]["coverage_mid"] = "/b"
        with self.assertRaisesRegex(JointPlanError, "coverage event is absent"):
            build_joint_materialization_plan(
                rows_by_split=rows,
                events_by_split=events,
                selected_labels={"A", "B"},
                selected_mids={"/a", "/b"},
                target_train_per_class=2,
                target_eval_per_class=1,
            )


if __name__ == "__main__":
    unittest.main()
