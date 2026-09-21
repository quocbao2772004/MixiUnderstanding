"""Tests for deterministic, identifier-only QCES-v5 development pilots."""

from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mixi_understanding.scripts.create_qces_v5_dev_pilot import (
    FORMAT,
    group_complete_families,
    main,
    select_families,
)


def rows(split: str, family_count: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for family_index in range(family_count):
        family_id = f"family_{split}_{family_index:06d}"
        for variant in ("base", "order_swap", "anchor_drop"):
            result.append(
                {
                    "id": f"{split}_{family_index:06d}_{variant}",
                    "split": split,
                    "scene_family_id": family_id,
                    "scene_id": f"scene_{split}_{family_index:06d}_{variant}",
                    "variant_id": variant,
                    "relation": "after",
                    "no_evidence": variant == "anchor_drop",
                    "source_group_ids": [f"source_{split}_{family_index:06d}"],
                }
            )
    return result


def write_jsonl(path: Path, items: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in items),
        encoding="utf-8",
    )


class DevPilotTest(unittest.TestCase):
    def test_identifier_selection_is_deterministic_and_counted(self) -> None:
        grouped = {f"family_train_{index:06d}": [] for index in range(20)}
        first = select_families(grouped, split="train", count=7, seed=2026)
        second = select_families(grouped, split="train", count=7, seed=2026)
        changed = select_families(grouped, split="train", count=7, seed=2027)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertNotEqual(first, changed)

    def test_incomplete_family_is_rejected(self) -> None:
        incomplete = rows("train", 1)[:-1]
        with self.assertRaisesRegex(ValueError, "variants"):
            group_complete_families(incomplete, manifest=Path("train.jsonl"))

    def test_main_writes_complete_dev_only_receipt_without_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            output_train = root / "pilot_train.jsonl"
            output_val = root / "pilot_val.jsonl"
            receipt = root / "pilot_receipt.json"
            write_jsonl(train, rows("train", 5))
            write_jsonl(val, rows("val", 4))

            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "--train-manifest",
                        str(train),
                        "--val-manifest",
                        str(val),
                        "--output-train",
                        str(output_train),
                        "--output-val",
                        str(output_val),
                        "--receipt",
                        str(receipt),
                        "--train-families",
                        "3",
                        "--val-families",
                        "2",
                        "--seed",
                        "11",
                    ]
                )

            payload = json.loads(receipt.read_text(encoding="utf-8"))
            pilot_train = [
                json.loads(line)
                for line in output_train.read_text(encoding="utf-8").splitlines()
            ]
            pilot_val = [
                json.loads(line)
                for line in output_val.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(payload["format"], FORMAT)
        self.assertFalse(payload["paper_result_eligible"])
        self.assertFalse(payload["test_split_accessed"])
        self.assertTrue(payload["selection"]["uses_identifiers_only"])
        self.assertFalse(payload["selection"]["uses_labels_answers_targets_or_metrics"])
        self.assertEqual(len(pilot_train), 9)
        self.assertEqual(len(pilot_val), 6)
        self.assertEqual(
            payload["isolation"]["train_val_source_group_overlap_count ↓"], 0
        )


if __name__ == "__main__":
    unittest.main()
