from __future__ import annotations

import unittest

from mixi_understanding.qces.availability_aware_acoustic_plan import (
    AvailabilityAwarePlanError,
    build_availability_aware_plan,
    refill_rejected_primary_rows,
)
from mixi_understanding.qces.supported_ontology import StrongEvent


def _event(video: str, label: str, mid: str, onset: float = 0.2) -> StrongEvent:
    return StrongEvent(
        segment_id=f"{video}_0",
        video_id=video,
        mid=mid,
        label=label,
        display_name=label,
        onset_seconds=onset,
        offset_seconds=onset + 0.4,
    )


def _clean(split: str, video: str, label: str, mid: str, tier: int = 0):
    return {
        "metadata_split": split,
        "video_id": video,
        "segment_id": f"{video}_0",
        "event_uid": f"event-{split}-{video}-{label}",
        "label": label,
        "mid": mid,
        "onset_seconds": 0.2,
        "offset_seconds": 0.6,
        "duration_seconds": 0.4,
        "clip_duration_seconds": 1.0,
        "ambiguity_tier": tier,
        "ambiguity_tier_name": f"tier-{tier}",
        "fully_isolated": tier == 0,
        "clean_source_eligible": tier <= 2,
        "selected_distractor_overlap_fraction": 0.0,
        "any_different_overlap_fraction": 0.0,
        "effective_min_isolation_margin_seconds": 0.2,
        "left_isolation_margin_seconds": 0.2,
        "right_isolation_margin_seconds": 0.2,
        "materialized": False,
        "materialized_audio_exists": False,
        "materialized_audio_path": "",
        "materialized_protocol_splits": [],
    }


def _available(
    split: str,
    video: str,
    labels: list[str],
    *,
    route: str = "preferred",
    row: int = 0,
):
    return {
        "source_route": route,
        "hf_dataset": "test/dataset",
        "hf_revision": "revision",
        "hf_split": "train" if split == "train" else "test",
        "video_id": video,
        "parquet_url": f"memory://{route}/{split}.parquet",
        "row_group": 0,
        "row_index": row,
        "labels": labels,
        "human_labels": [],
        "shard_provenance": {"lfs_sha256": "a" * 64},
    }


class AvailabilityAwareAcousticPlanTests(unittest.TestCase):
    def _fixtures(self):
        selected = {"A": "/a", "B": "/b"}
        clean = []
        availability = []
        events = {"train": {}, "eval": {}}
        row = 0
        for split, videos in (
            ("train", (("ta1", "A", "/a"), ("ta2", "A", "/a"),
                       ("tb1", "B", "/b"), ("tb2", "B", "/b"))),
            ("eval", (("ea1", "A", "/a"), ("eb1", "B", "/b"))),
        ):
            for video, label, mid in videos:
                clean.append(_clean(split, video, label, mid))
                availability.append(_available(split, video, [mid], row=row))
                events[split][video] = [_event(video, label, mid)]
                row += 1
        # A lower-priority duplicate mirror must remain auditable but not be
        # selected over the configured preferred route.
        availability.append(
            _available("train", "ta1", ["/a"], route="fallback", row=99)
        )
        return selected, clean, availability, events

    def test_exact_quota_is_selected_after_strict_availability_join(self) -> None:
        selected, clean, availability, events = self._fixtures()
        plans, report = build_availability_aware_plan(
            cleanliness_rows=clean,
            availability_entries=availability,
            events_by_split=events,
            selected_label_to_mid=selected,
            route_priority=("preferred", "fallback"),
            target_train_videos_per_class=2,
            target_eval_videos_per_class=1,
            crop_padding_seconds=0.1,
        )
        self.assertTrue(report["audit_passes"])
        self.assertEqual(len(plans["train"]), 4)
        self.assertEqual(len(plans["eval"]), 2)
        self.assertEqual(report["deficits"], [])
        ta1 = next(row for row in plans["train"] if row["video_id"] == "ta1")
        self.assertEqual(ta1["materialization_source_route"], "preferred")
        self.assertTrue(ta1["availability_route_label_verified"])
        self.assertEqual(report["train_eval_source_overlap"], 0)
        self.assertEqual(report["selection_key_count"], 6)

    def test_label_mismatch_is_excluded_and_reported_as_real_deficit(self) -> None:
        selected, clean, availability, events = self._fixtures()
        for row in availability:
            if row["video_id"] == "tb2":
                row["labels"] = ["/wrong"]
        plans, report = build_availability_aware_plan(
            cleanliness_rows=clean,
            availability_entries=availability,
            events_by_split=events,
            selected_label_to_mid=selected,
            route_priority=("preferred", "fallback"),
            target_train_videos_per_class=2,
            target_eval_videos_per_class=1,
            require_label_identity=True,
        )
        self.assertFalse(report["audit_passes"])
        self.assertEqual(len(plans["train"]), 3)
        deficit = next(
            row
            for row in report["deficits"]
            if row["metadata_split"] == "train" and row["label"] == "B"
        )
        self.assertEqual(deficit["deficit"], 1)
        self.assertGreater(
            report["availability"]["label_mismatched_locations"], 0
        )

    def test_cross_official_split_source_overlap_is_rejected(self) -> None:
        selected, clean, availability, events = self._fixtures()
        eval_a = next(row for row in clean if row["video_id"] == "ea1")
        eval_a["video_id"] = "ta1"
        eval_a["segment_id"] = "ta1_0"
        eval_a["event_uid"] = "event-eval-ta1-A"
        availability.append(_available("eval", "ta1", ["/a"], row=100))
        events["eval"]["ta1"] = [_event("ta1", "A", "/a")]
        with self.assertRaisesRegex(
            AvailabilityAwarePlanError, "train/eval source overlap"
        ):
            build_availability_aware_plan(
                cleanliness_rows=clean,
                availability_entries=availability,
                events_by_split=events,
                selected_label_to_mid=selected,
                route_priority=("preferred", "fallback"),
                target_train_videos_per_class=2,
                target_eval_videos_per_class=1,
            )

    def test_one_canonical_physical_location_covers_multilabel_video(self) -> None:
        selected = {"A": "/a", "B": "/b"}
        clean = [
            _clean("train", "multi", "A", "/a"),
            _clean("train", "multi", "B", "/b"),
            _clean("eval", "eval-a", "A", "/a"),
            _clean("eval", "eval-b", "B", "/b"),
        ]
        availability = [
            _available("train", "multi", ["/a"], route="preferred", row=0),
            _available(
                "train", "multi", ["/a", "/b"], route="fallback", row=1
            ),
            _available("eval", "eval-a", ["/a"], row=2),
            _available("eval", "eval-b", ["/b"], row=3),
        ]
        events = {
            "train": {
                "multi": [
                    _event("multi", "A", "/a", onset=0.1),
                    _event("multi", "B", "/b", onset=0.6),
                ]
            },
            "eval": {
                "eval-a": [_event("eval-a", "A", "/a")],
                "eval-b": [_event("eval-b", "B", "/b")],
            },
        }
        plans, report = build_availability_aware_plan(
            cleanliness_rows=clean,
            availability_entries=availability,
            events_by_split=events,
            selected_label_to_mid=selected,
            route_priority=("preferred", "fallback"),
            target_train_videos_per_class=1,
            target_eval_videos_per_class=1,
            reserve_train_videos_per_class=0,
            reserve_eval_videos_per_class=0,
        )
        multi_rows = [row for row in plans["train"] if row["video_id"] == "multi"]
        self.assertEqual(len(multi_rows), 2)
        self.assertEqual(
            {row["materialization_source_route"] for row in multi_rows},
            {"fallback"},
        )
        self.assertEqual(
            {
                (
                    row["availability_location"]["parquet_url"],
                    row["availability_location"]["row_group"],
                    row["availability_location"]["row_index"],
                )
                for row in multi_rows
            },
            {("memory://fallback/train.parquet", 0, 1)},
        )
        self.assertTrue(
            report["invariants"]["one_physical_location_per_official_split_video"]
        )

    def test_frozen_reserve_refills_rejection_without_source_overlap(self) -> None:
        selected = {"A": "/a", "B": "/b"}
        clean = []
        availability = []
        events = {"train": {}, "eval": {}}
        row_index = 0
        for split, count in (("train", 3), ("eval", 2)):
            for label, mid in selected.items():
                for index in range(count):
                    video = f"{split}-{label.lower()}-{index}"
                    clean.append(_clean(split, video, label, mid, tier=index))
                    availability.append(
                        _available(split, video, [mid], row=row_index)
                    )
                    events[split][video] = [_event(video, label, mid)]
                    row_index += 1
        plans, report = build_availability_aware_plan(
            cleanliness_rows=clean,
            availability_entries=availability,
            events_by_split=events,
            selected_label_to_mid=selected,
            route_priority=("preferred",),
            target_train_videos_per_class=1,
            target_eval_videos_per_class=1,
            reserve_train_videos_per_class=1,
            reserve_eval_videos_per_class=1,
        )
        primary_sources = {
            row["video_id"] for row in [*plans["train"], *plans["eval"]]
        }
        reserve_sources = {
            row["video_id"]
            for row in [*plans["reserve_train"], *plans["reserve_eval"]]
        }
        self.assertFalse(primary_sources & reserve_sources)
        self.assertEqual(report["primary_reserve_source_overlap"], 0)
        rejected = next(
            row for row in plans["train"] if row["coverage_label"] == "A"
        )
        refilled, refill_report = refill_rejected_primary_rows(
            primary_by_split={"train": plans["train"], "eval": plans["eval"]},
            reserve_by_split={
                "train": plans["reserve_train"],
                "eval": plans["reserve_eval"],
            },
            rejected_selection_keys={str(rejected["selection_key"])},
        )
        self.assertTrue(refill_report["audit_passes"])
        self.assertEqual(refill_report["replacement_rows"], 1)
        self.assertEqual(len(refilled["train"]), len(plans["train"]))
        replacement = next(
            row
            for row in refilled["train"]
            if row.get("replaced_primary_selection_key")
            == rejected["selection_key"]
        )
        self.assertNotIn(replacement["video_id"], primary_sources)


if __name__ == "__main__":
    unittest.main()
