from __future__ import annotations

import unittest

from mixi_understanding.qces.reserve_quality_topup import (
    ReserveTopupError,
    build_joint_topup_plan,
    build_reserve_quality_topup,
    collect_quality_outcomes,
    constant_targets,
    flatten_crop_rows,
    targets_from_primary_contract,
)
from mixi_understanding.qces.supported_ontology import StrongEvent


PROTOCOL = "fixed_audiosep_gate_v1"


def crop(
    key: str,
    label: str,
    split: str,
    video: str,
    *,
    partition: str,
    tier: int = 0,
    rank: int = 1,
    mid: str | None = None,
) -> dict:
    hf_split = "train" if split == "train" else "test"
    mid = mid or f"/m/{label.lower()}"
    route = f"route_{split}"
    dataset = f"dataset_{split}"
    revision = f"revision_{split}"
    row = {
        "format": "qces_availability_aware_acoustic_plan_v1",
        "selection_key": key,
        "selection_partition": partition,
        "metadata_split": split,
        "video_id": video,
        "segment_id": f"{video}_0",
        "coverage_label": label,
        "coverage_mid": mid,
        "coverage_rank": rank,
        "ambiguity_tier": tier,
        "ambiguity_tier_name": f"tier_{tier}",
        "crop_start_seconds": 0.5,
        "crop_end_seconds": 2.5,
        "event_onset_seconds": 1.0,
        "event_offset_seconds": 2.0,
        "split_lock": "unassigned_train_pool" if split == "train" else "test",
        "materialization_source_route": route,
        "materialization_hf_dataset": dataset,
        "materialization_hf_revision": revision,
        "materialization_hf_split": hf_split,
        "availability_location": {
            "source_route": route,
            "hf_dataset": dataset,
            "hf_revision": revision,
            "hf_split": hf_split,
            "video_id": video,
            "parquet_url": f"hf://{dataset}/{video}.parquet",
            "row_group": 0,
            "row_index": rank,
            "labels": [mid],
        },
    }
    if partition == "reserve":
        row["reserve_rank"] = rank
    return row


def outcome(key: str, accepted: bool, tier: str | None = None) -> dict:
    return {
        "item_id": key,
        "accepted": accepted,
        "acceptance_tier": tier or ("gold" if accepted else "rejected"),
        "quality_protocol": PROTOCOL,
    }


def event(row: dict) -> StrongEvent:
    return StrongEvent(
        segment_id=row["segment_id"],
        video_id=row["video_id"],
        mid=row["coverage_mid"],
        label=row["coverage_label"],
        display_name=row["coverage_label"].replace("_", " "),
        onset_seconds=row["event_onset_seconds"],
        offset_seconds=row["event_offset_seconds"],
    )


def event_index(rows: list[dict]) -> dict[str, dict[str, list[StrongEvent]]]:
    output: dict[str, dict[str, list[StrongEvent]]] = {"train": {}, "eval": {}}
    for row in rows:
        output[row["metadata_split"]].setdefault(row["video_id"], []).append(event(row))
    return output


class ReserveQualityTopupTest(unittest.TestCase):
    def test_overdraw_two_prefers_lower_ambiguity_then_frozen_rank(self) -> None:
        p_train = [
            crop("pt1", "A", "train", "ptv1", partition="primary", rank=1),
            crop("pt2", "A", "train", "ptv2", partition="primary", rank=2),
        ]
        p_eval = [crop("pe1", "A", "eval", "pev1", partition="primary")]
        r_train = [
            crop("rt-tier1", "A", "train", "rtv1", partition="reserve", tier=1, rank=1),
            crop("rt-rank9", "A", "train", "rtv2", partition="reserve", tier=0, rank=9),
            crop("rt-rank3", "A", "train", "rtv3", partition="reserve", tier=0, rank=3),
        ]
        r_eval = [
            crop("re-rank2", "A", "eval", "rev2", partition="reserve", tier=3, rank=2),
            crop("re-rank1", "A", "eval", "rev1", partition="reserve", tier=3, rank=1),
        ]
        artifacts, receipt = build_reserve_quality_topup(
            primary_by_split={"train": p_train, "eval": p_eval},
            reserve_by_split={"train": r_train, "eval": r_eval},
            quality_rows=[outcome("pt1", True), outcome("pt2", False), outcome("pe1", False)],
            targets_by_split_label={"train": {"A": 2}, "eval": {"A": 1}},
            events_by_split=event_index([*r_train, *r_eval]),
            overdraw_factor=2,
        )
        self.assertTrue(receipt["audit_passes"])
        self.assertEqual(
            [row["selection_key"] for row in artifacts["pending_reserve_requests"]["train"]],
            ["rt-rank3", "rt-rank9"],
        )
        self.assertEqual(
            [row["selection_key"] for row in artifacts["pending_reserve_requests"]["eval"]],
            ["re-rank1", "re-rank2"],
        )
        self.assertEqual(receipt["pending_class_video_rows"], 4)
        self.assertEqual(receipt["pending_joint_source_video_rows"], 4)
        self.assertFalse(receipt["invariants"]["qa_or_detector_outcomes_used_for_selection"])

    def test_consumes_terminal_reserve_prefix_then_emits_next_wave(self) -> None:
        primary = [
            crop("p1", "A", "train", "pv1", partition="primary", rank=1),
            crop("p2", "A", "train", "pv2", partition="primary", rank=2),
        ]
        primary_eval = [crop("pe", "A", "eval", "pev", partition="primary")]
        reserves = [
            crop("r1", "A", "train", "rv1", partition="reserve", rank=1),
            crop("r2", "A", "train", "rv2", partition="reserve", rank=2),
            crop("r3", "A", "train", "rv3", partition="reserve", rank=3),
            crop("r4", "A", "train", "rv4", partition="reserve", rank=4),
        ]
        reserve_eval = [crop("re", "A", "eval", "rev", partition="reserve")]
        artifacts, _ = build_reserve_quality_topup(
            primary_by_split={"train": primary, "eval": primary_eval},
            reserve_by_split={"train": reserves, "eval": reserve_eval},
            quality_rows=[
                outcome("p1", False),
                outcome("p2", False),
                outcome("pe", True),
                outcome("r1", False),
                outcome("r2", True, "silver"),
            ],
            targets_by_split_label={"train": {"A": 2}, "eval": {"A": 1}},
            events_by_split=event_index([*reserves, *reserve_eval]),
            overdraw_factor=2,
        )
        self.assertEqual(
            [row["selection_key"] for row in artifacts["accepted_reserve_topups"]["train"]],
            ["r2"],
        )
        self.assertEqual(
            [row["selection_key"] for row in artifacts["pending_reserve_requests"]["train"]],
            ["r3", "r4"],
        )

    def test_joint_builder_merges_two_class_requests_on_one_video(self) -> None:
        one = crop("a", "A", "train", "shared", partition="reserve", rank=1)
        two = crop(
            "b", "B", "train", "shared", partition="reserve", rank=1, mid="/m/b"
        )
        # One physical source must have one immutable location even when it
        # supports two event labels.
        two["availability_location"] = dict(one["availability_location"])
        two["coverage_mid"] = "/m/b"
        events = event_index([one, two])
        joint = build_joint_topup_plan(
            pending_by_split={"train": [two, one], "eval": []},
            events_by_split=events,
            selected_mids={"/m/a", "/m/b"},
        )
        self.assertEqual(len(joint["train"]), 1)
        self.assertEqual(joint["train"][0]["coverage_request_count"], 2)
        self.assertEqual(joint["train"][0]["covers_deficit_labels"], ["A", "B"])
        self.assertTrue(joint["train"][0]["full_official_strong_events"])

    def test_rejects_nonprefix_reserve_evaluation(self) -> None:
        p_train = [crop("p", "A", "train", "pv", partition="primary")]
        p_eval = [crop("pe", "A", "eval", "pev", partition="primary")]
        reserves = [
            crop("r1", "A", "train", "rv1", partition="reserve", rank=1),
            crop("r2", "A", "train", "rv2", partition="reserve", rank=2),
        ]
        reserve_eval = [crop("re", "A", "eval", "rev", partition="reserve")]
        with self.assertRaisesRegex(ReserveTopupError, "skipped earlier frozen"):
            build_reserve_quality_topup(
                primary_by_split={"train": p_train, "eval": p_eval},
                reserve_by_split={"train": reserves, "eval": reserve_eval},
                quality_rows=[outcome("p", False), outcome("pe", True), outcome("r2", True)],
                targets_by_split_label={"train": {"A": 1}, "eval": {"A": 1}},
                events_by_split=event_index([*reserves, *reserve_eval]),
            )

    def test_rejects_primary_reserve_source_overlap(self) -> None:
        p_train = [crop("p", "A", "train", "same", partition="primary")]
        p_eval = [crop("pe", "A", "eval", "pev", partition="primary")]
        r_train = [crop("r", "A", "train", "same", partition="reserve")]
        r_eval = [crop("re", "A", "eval", "rev", partition="reserve")]
        with self.assertRaisesRegex(ReserveTopupError, "source-disjoint"):
            build_reserve_quality_topup(
                primary_by_split={"train": p_train, "eval": p_eval},
                reserve_by_split={"train": r_train, "eval": r_eval},
                quality_rows=[outcome("p", False), outcome("pe", True)],
                targets_by_split_label={"train": {"A": 1}, "eval": {"A": 1}},
                events_by_split=event_index([r_train[0], r_eval[0]]),
            )

    def test_quality_gate_is_boolean_and_protocol_locked(self) -> None:
        duplicated = [outcome("x", True, "silver"), outcome("x", True, "silver")]
        values, protocols = collect_quality_outcomes(duplicated)
        self.assertTrue(values["x"]["accepted"])
        self.assertEqual(protocols, {PROTOCOL})
        wrong = outcome("x", True, "rejected")
        with self.assertRaisesRegex(ReserveTopupError, "inconsistent"):
            collect_quality_outcomes([wrong])
        mixed = outcome("y", True)
        mixed["quality_protocol"] = "changed_thresholds"
        with self.assertRaisesRegex(ReserveTopupError, "mix fixed protocols"):
            collect_quality_outcomes([outcome("x", True), mixed])

    def test_flatten_and_target_helpers_support_joint_smoke_contract(self) -> None:
        train = crop("p", "A", "train", "pv", partition="primary")
        eval_row = crop("e", "A", "eval", "ev", partition="primary")
        joint = {"metadata_split": "train", "video_id": "pv", "crop_requests": [train]}
        self.assertEqual(flatten_crop_rows([joint]), [train])
        primary = {"train": [joint], "eval": [eval_row]}
        self.assertEqual(
            targets_from_primary_contract(primary),
            {"train": {"A": 1}, "eval": {"A": 1}},
        )
        self.assertEqual(
            constant_targets(primary, train_target=100, eval_target=20),
            {"train": {"A": 100}, "eval": {"A": 20}},
        )


if __name__ == "__main__":
    unittest.main()
