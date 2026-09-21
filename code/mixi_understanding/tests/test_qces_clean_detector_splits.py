from __future__ import annotations

from typing import Any
import unittest

from mixi_understanding.qces.clean_detector_splits import (
    SplitProtocolError,
    build_clean_detector_splits,
)


def _audioset_row(
    video_id: str,
    labels: str | tuple[str, ...],
    *,
    split: str,
    suffix: str = "0",
) -> dict[str, Any]:
    labels = (labels,) if isinstance(labels, str) else labels
    return {
        "scene_id": f"as_{split}_{video_id}_{suffix}",
        "video_id": video_id,
        "hf_split": split,
        "audio_sha256": f"audio-{split}-{video_id}-{suffix}",
        "mixture_path": f"audio/{split}/{video_id}-{suffix}.wav",
        "events": [
            {
                "event_id": f"event-{split}-{video_id}-{suffix}-{index}",
                "event_kind": "semantic",
                "label": label,
                "onset_seconds": float(index),
                "offset_seconds": float(index + 1),
            }
            for index, label in enumerate(labels)
        ],
    }


def _preserved_row(source_id: str, label: str, *, split: str) -> dict[str, Any]:
    return {
        "scene_id": f"preserved-{split}-{source_id}",
        "source_id": source_id,
        "split": split,
        "mixture_path": f"preserved/{split}/{source_id}.wav",
        "events": [
            {
                "event_id": f"preserved-event-{split}-{source_id}",
                "event_kind": "semantic",
                "label": label,
                "onset_seconds": 0.1,
                "offset_seconds": 0.9,
                "source_id": source_id,
                "source_sha256": f"source-sha-{split}-{source_id}",
                "source_path": f"source/{split}/{source_id}.wav",
            }
        ],
    }


def _preserved() -> dict[str, list[dict[str, Any]]]:
    return {
        "train": [_preserved_row("p-train", "A", split="train")],
        "dev": [_preserved_row("p-dev", "B", split="validation")],
        "test": [_preserved_row("p-test", "A", split="eval")],
    }


class CleanDetectorSplitTests(unittest.TestCase):
    def test_clean_split_is_deterministic_grouped_and_test_only(self) -> None:
        train_rows = [
            _audioset_row(
                f"v{index:02d}",
                ("A", "Rare") if index == 0 else ("A" if index < 6 else "B"),
                split="train",
            )
            for index in range(12)
        ]
        # A second row from one video proves rows are grouped rather than split
        # independently.
        train_rows.append(_audioset_row("v05", "B", split="train", suffix="1"))
        test_rows = [_audioset_row("official-test", "B", split="test")]

        first = build_clean_detector_splits(
            audioset_train_rows=train_rows,
            audioset_test_rows=test_rows,
            preserved_rows=_preserved(),
            ontology=["A", "B", "Rare"],
            dev_fraction=0.25,
            seed=17,
        )
        second = build_clean_detector_splits(
            audioset_train_rows=list(reversed(train_rows)),
            audioset_test_rows=list(reversed(test_rows)),
            preserved_rows={
                split: list(reversed(rows)) for split, rows in _preserved().items()
            },
            ontology=["A", "B", "Rare"],
            dev_fraction=0.25,
            seed=17,
        )
        self.assertEqual(first.splits, second.splits)
        self.assertTrue(first.receipt["identity_audit"]["passes"])
        self.assertEqual(first.receipt["stratification"]["actual_dev_groups"], 3)

        split_by_video = {
            row["video_id"]: split
            for split, rows in first.splits.items()
            for row in rows
            if row["protocol_source"] == "audioset_strong"
        }
        self.assertEqual(split_by_video["official-test"], "test")
        self.assertEqual(split_by_video["v00"], "train")
        self.assertEqual(
            {
                row["protocol_split"]
                for rows in first.splits.values()
                for row in rows
                if row.get("video_id") == "v05"
            },
            {split_by_video["v05"]},
        )
        self.assertEqual(
            {
                row["source_id"]: row["protocol_split"]
                for rows in first.splits.values()
                for row in rows
                if row["protocol_source"] == "preserved_official"
            },
            {"p-train": "train", "p-dev": "dev", "p-test": "test"},
        )
        self.assertEqual(
            first.receipt["invariants"],
            {
                "official_test_rows_outside_test": 0,
                "preserved_assignment_violations": 0,
                "cross_split_identity_overlaps": 0,
                "ontology_train_positive_labels": 3,
                "ontology_missing_train_positive_labels": [],
            },
        )

    def test_rejects_audioset_video_overlap_between_official_splits(self) -> None:
        with self.assertRaisesRegex(SplitProtocolError, "share video_id"):
            build_clean_detector_splits(
                audioset_train_rows=[_audioset_row("same", "A", split="train")],
                audioset_test_rows=[_audioset_row("same", "A", split="test")],
                preserved_rows={"train": [], "dev": [], "test": []},
                ontology=["A"],
                dev_fraction=0.0,
            )

    def test_rejects_preserved_source_identity_leakage(self) -> None:
        preserved = _preserved()
        preserved["dev"][0]["events"][0]["label"] = "A"
        preserved["dev"][0]["events"][0]["source_sha256"] = preserved["train"][0][
            "events"
        ][0]["source_sha256"]
        with self.assertRaisesRegex(SplitProtocolError, "identity overlap"):
            build_clean_detector_splits(
                audioset_train_rows=[_audioset_row("train-a", "A", split="train")],
                audioset_test_rows=[_audioset_row("test-a", "A", split="test")],
                preserved_rows=preserved,
                ontology=["A"],
                dev_fraction=0.0,
            )

    def test_rejects_preserved_manifest_in_wrong_partition(self) -> None:
        preserved = _preserved()
        preserved["dev"][0]["split"] = "test"
        with self.assertRaisesRegex(SplitProtocolError, "declares split"):
            build_clean_detector_splits(
                audioset_train_rows=[_audioset_row("train-a", "A", split="train")],
                audioset_test_rows=[_audioset_row("test-a", "A", split="test")],
                preserved_rows=preserved,
                ontology=["A"],
                dev_fraction=0.0,
            )

    def test_rejects_ontology_without_train_positive(self) -> None:
        with self.assertRaisesRegex(
            SplitProtocolError, "without a positive training scene"
        ):
            build_clean_detector_splits(
                audioset_train_rows=[_audioset_row("train-a", "A", split="train")],
                audioset_test_rows=[_audioset_row("test-b", "B", split="test")],
                preserved_rows={"train": [], "dev": [], "test": []},
                ontology=["A", "B"],
                dev_fraction=0.0,
            )


if __name__ == "__main__":
    unittest.main()
