"""Synthetic-metadata tests for the development-only pseudo-label-OOD planner."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from unittest import mock

from mixi_understanding.scripts.plan_qces_v5_pseudo_label_ood import (
    ALGORITHM_VERSION,
    FOLD_COUNT,
    ManifestReceipt,
    _read_manifest,
    _role_or_absent_labels,
    _semantic_event_labels,
    _validate_manifest_path,
    make_plan,
    plan_manifests,
)


@dataclass(frozen=True)
class FakeEvent:
    event_id: str
    label: str
    event_kind: str = "semantic"


@dataclass(frozen=True)
class FakeRecord:
    sample_id: str
    split: str
    scene_family_id: str
    scene_id: str
    variant_id: str
    primary_counterfactual_probe: bool
    counterfactual_group_id: str
    question_semantics_id: str
    question: str
    answer_options: Tuple[str, ...]
    surface_control_group_id: Optional[str]
    mention_order_variant: str
    events: Tuple[FakeEvent, ...]
    no_evidence: bool
    absent_labels: Tuple[str, ...]
    anchor_event_ids: Tuple[str, ...]
    answer_event_ids: Tuple[str, ...]
    relation: str
    same_label_repeat: bool
    semantic_overlap: bool


VARIANTS = ("base", "order_swap", "anchor_drop")
FAMILY_LABELS = ("Alpha", "Beta", "Gamma", None)


def _events(family_label: Optional[str]) -> Tuple[FakeEvent, ...]:
    labels = ["Repeat", "Repeat", family_label or "Neutral", "Common A", "Common B"]
    return tuple(
        FakeEvent(event_id=f"e{index}", label=label)
        for index, label in enumerate(labels)
    )


def _record(
    *, split: str, family_index: int, family_label: Optional[str],
    variant: str, question_index: int,
) -> FakeRecord:
    events = _events(family_label)
    relation = ("after", "before", "first")[question_index % 3]
    no_evidence = False
    absent_labels: Tuple[str, ...] = ()
    anchor_ids: Tuple[str, ...] = ("e0",)
    answer_ids: Tuple[str, ...] = ("e1",)

    # A heldout label is a positive role probe in each relation in its own
    # family.  Index 0 is also the one-per-variant primary CEE row.
    if family_label is not None and question_index in {0, 1, 2, 3}:
        relation = {0: "after", 1: "after", 2: "before", 3: "first"}[
            question_index
        ]
        anchor_ids = ("e2",)
        answer_ids = ("e3",)

    # A separate family supplies genuine no-evidence probes.  This exercises
    # record-level removal of a label surface after acoustic families are gone.
    if family_label is None and question_index in {4, 5, 6}:
        heldout = ("Alpha", "Beta", "Gamma")[question_index - 4]
        relation = ("after", "before", "first")[question_index - 4]
        no_evidence = True
        absent_labels = (heldout,)
        anchor_ids = ()
        answer_ids = ()

    family_id = f"family_{split}_{family_index:06d}"
    scene_id = f"scene_{split}_{family_index:06d}_{variant}"
    return FakeRecord(
        sample_id=f"{split}_{family_index:06d}_{variant}_{question_index:02d}",
        split=split,
        scene_family_id=family_id,
        scene_id=scene_id,
        variant_id=variant,
        primary_counterfactual_probe=question_index == 0,
        counterfactual_group_id=f"{family_id}:primary",
        question_semantics_id=f"{family_id}:primary-semantics",
        question=f"Primary question for {family_id}",
        answer_options=("Alpha", "Beta", "Gamma", "Common A", "no_evidence"),
        surface_control_group_id=None,
        mention_order_variant="not_applicable",
        events=events,
        no_evidence=no_evidence,
        absent_labels=absent_labels,
        anchor_event_ids=anchor_ids,
        answer_event_ids=answer_ids,
        relation=relation,
        same_label_repeat=True,
        semantic_overlap=True,
    )


def _split_records(split: str) -> list[FakeRecord]:
    return [
        _record(
            split=split,
            family_index=family_index,
            family_label=family_label,
            variant=variant,
            question_index=question_index,
        )
        for family_index, family_label in enumerate(FAMILY_LABELS)
        for variant in VARIANTS
        for question_index in range(16)
    ]


def _receipt(split: str, records: list[FakeRecord]) -> ManifestReceipt:
    return ManifestReceipt(
        path=f"/synthetic/qces_{split}.jsonl",
        sha256=("1" if split == "train" else "2") * 64,
        bytes=1234,
        records=len(records),
        scene_families=len(FAMILY_LABELS),
        record_ids_sha256=("3" if split == "train" else "4") * 64,
        accepted_split=split,
    )


class PseudoLabelOODPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = _split_records("train")
        self.val = _split_records("val")

    def _plan(self, **kwargs: object) -> dict:
        arguments = {
            "train_records": self.train,
            "val_records": self.val,
            "train_receipt": _receipt("train", self.train),
            "val_receipt": _receipt("val", self.val),
            "seed": 2718,
            "labels_per_fold": 1,
        }
        arguments.update(kwargs)
        return make_plan(**arguments)  # type: ignore[arg-type]

    def test_three_folds_are_deterministic_disjoint_and_coverage_gated(self) -> None:
        first = self._plan()
        second = self._plan()
        self.assertEqual(first, second)
        self.assertEqual(first["algorithm"]["version"], ALGORITHM_VERSION)
        self.assertEqual(len(first["folds"]), FOLD_COUNT)
        selected = [
            label
            for fold in first["folds"]
            for label in fold["heldout_labels"]
        ]
        self.assertEqual(set(selected), {"Alpha", "Beta", "Gamma"})
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(first["global_proof"]["only_train_and_val_records_loaded"])
        for fold in first["folds"]:
            self.assertTrue(fold["coverage"]["passed"])
            counts = fold["counts"]["heldout_probes"]
            self.assertGreaterEqual(counts["records"], 12)
            self.assertTrue(all(counts["by_relation"].values()))
            self.assertGreater(counts["by_no_evidence"]["true"], 0)
            self.assertGreater(counts["by_same_label_repeat"]["true"], 0)
            self.assertGreater(counts["by_semantic_overlap"]["true"], 0)

    def test_meta_train_has_no_acoustic_or_controller_label_exposure(self) -> None:
        plan = self._plan()
        train_by_id = {record.sample_id: record for record in self.train}
        for fold in plan["folds"]:
            heldout = set(fold["heldout_labels"])
            self.assertTrue(fold["leakage_proof"]["passed"])
            for record_id in fold["meta_train_ids"]:
                record = train_by_id[record_id]
                self.assertFalse(_semantic_event_labels(record) & heldout)
                self.assertFalse(_role_or_absent_labels(record) & heldout)
            for family_id in fold["excluded_acoustic_family_ids"]:
                self.assertNotIn(family_id, fold["meta_train_family_ids"])
            self.assertEqual(len(fold["excluded_controller_exposure_ids"]), 3)

    def test_probe_ids_use_exact_role_or_absent_rule_and_keep_full_families(self) -> None:
        plan = self._plan()
        val_by_id = {record.sample_id: record for record in self.val}
        family_ids = {
            record.scene_family_id: {
                item.sample_id
                for item in self.val
                if item.scene_family_id == record.scene_family_id
            }
            for record in self.val
        }
        for fold in plan["folds"]:
            heldout = set(fold["heldout_labels"])
            expected_probes = {
                record.sample_id
                for record in self.val
                if _role_or_absent_labels(record) & heldout
            }
            self.assertEqual(set(fold["heldout_probe_ids"]), expected_probes)
            expected_families = {
                val_by_id[record_id].scene_family_id
                for record_id in expected_probes
            }
            expected_full_ids = set().union(
                *(family_ids[family_id] for family_id in expected_families)
            )
            self.assertEqual(
                set(fold["meta_validation_family_ids"]), expected_families
            )
            self.assertEqual(set(fold["meta_validation_ids"]), expected_full_ids)
            self.assertTrue(expected_probes.issubset(expected_full_ids))

    def test_incomplete_validation_family_fails_closed(self) -> None:
        incomplete = self.val[:-1]
        with self.assertRaisesRegex(ValueError, "incomplete QCES-v5 family"):
            self._plan(
                val_records=incomplete,
                val_receipt=_receipt("val", incomplete),
            )

    def test_missing_no_evidence_coverage_fails_closed(self) -> None:
        stripped = [
            record
            if not record.no_evidence
            else FakeRecord(
                **{
                    **record.__dict__,
                    "no_evidence": False,
                    "absent_labels": (),
                    "anchor_event_ids": ("e0",),
                    "answer_event_ids": ("e1",),
                }
            )
            for record in self.val
        ]
        with self.assertRaisesRegex(ValueError, "coverage minimums"):
            self._plan(
                val_records=stripped,
                val_receipt=_receipt("val", stripped),
            )


class PseudoLabelOODInputGateTest(unittest.TestCase):
    def test_test_named_path_is_rejected_before_json_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qces_test_label_ood.jsonl"
            path.write_text("this is deliberately not JSON\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test-like manifest path"):
                _validate_manifest_path(path, "train")

    def test_renamed_validation_split_is_not_accepted_as_train(self) -> None:
        fake = _record(
            split="val",
            family_index=0,
            family_label="Alpha",
            variant="base",
            question_index=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qces_train.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with mock.patch(
                "mixi_understanding.scripts.plan_qces_v5_pseudo_label_ood."
                "parse_qces_v5_record",
                return_value=fake,
            ):
                with self.assertRaisesRegex(ValueError, "only 'train' is accepted"):
                    _read_manifest(path, "train")

    def test_output_cannot_overwrite_or_rewrite_an_input_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "qces_train.jsonl"
            val = root / "qces_val.jsonl"
            train.write_text("{}\n", encoding="utf-8")
            val.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                plan_manifests(
                    train_manifest=train,
                    val_manifest=val,
                    output=train,
                    labels_per_fold=1,
                    overwrite=True,
                )


if __name__ == "__main__":
    unittest.main()
