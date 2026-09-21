"""Synthetic-fixture tests for the strict FUSS/FSD50K source gate."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

from mixi_understanding.data.qces_v5_source_ledger import (
    COMPLIANCE_FORMAT,
    PINS_FORMAT,
    PLAN_FORMAT,
    RECEIPT_FORMAT,
    SELECTION_FORMAT,
    SourceAuditError,
    audit_source_ledger,
    fsd50k_uploader_key,
    sha256_file,
)
from mixi_understanding.scripts.audit_qces_v5_fuss_sources import (
    BLOCKED_RECEIPT_FORMAT,
    RECEIPT_NAME,
    main as audit_cli_main,
)


EXPECTED_PARTITION_COUNTS = {
    "train": 136,
    "val": 68,
    "test_iid": 68,
    "test_compositional_ood": 68,
    "test_label_ood": 48,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class SyntheticLedger:
    def __init__(self, root: Path) -> None:
        self.project_root = root / "project"
        self.fuss_root = self.project_root / "upstream" / "fuss_v1.3"
        self.fsd50k_root = self.project_root / "upstream" / "fsd50k_v1.0"
        self.fuss_manifest = self.fuss_root / "metadata" / "fuss_sources.jsonl"
        self.fsd_manifest = (
            self.fsd50k_root / "metadata" / "fsd50k_labels_provenance.jsonl"
        )
        self.fuss_license = self.fuss_root / "LICENSE"
        self.fsd_license = self.fsd50k_root / "LICENSE"
        self.pins_path = self.project_root / "qces_v5_source_pins.json"
        self.selection_path = self.project_root / "qces_v5_source_selection.json"
        self.fuss_rows: List[Dict[str, Any]] = []
        self.fsd_rows: List[Dict[str, Any]] = []
        self._next_id = 100_000
        self._make_rows()
        self._write_all()

    def _add(self, *, label: str, split: str, duration: float) -> None:
        source_id = str(self._next_id)
        self._next_id += 1
        relative_audio = f"audio/{source_id}.wav"
        payload = b"synthetic-test-audio-v1\0" + source_id.encode("ascii")
        audio_path = self.fuss_root / relative_audio
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.fuss_rows.append(
            {
                "source_id": source_id,
                "split": split,
                "audio_path": relative_audio,
                "sha256": digest,
                "duration_seconds": duration,
                "source_license_spdx": "CC0-1.0",
                "source_license_url": (
                    "https://creativecommons.org/publicdomain/zero/1.0/"
                ),
                "dataset_version": "1.3",
            }
        )
        self.fsd_rows.append(
            {
                "source_id": source_id,
                "split": split,
                "labels": [label],
                "creator_id": fsd50k_uploader_key(f"uploader-{source_id}"),
                "uploader_id": fsd50k_uploader_key(f"uploader-{source_id}"),
                "uploader_name": f"uploader-{source_id}",
                "attribution": f"Synthetic creator {source_id}",
                "source_license_spdx": "CC0-1.0",
                "source_license_url": (
                    "https://creativecommons.org/publicdomain/zero/1.0/"
                ),
                "dataset_version": "1.0",
            }
        )

    def _make_rows(self) -> None:
        self.seen_labels = [f"seen_{index:02d}" for index in range(30)]
        self.heldout_labels = [f"heldout_{index:02d}" for index in range(10)]
        self.nuisance_labels = [f"nuisance_{index:02d}" for index in range(8)]
        for label in self.seen_labels:
            for split, count in (("train", 4), ("validation", 2), ("eval", 4)):
                for _ in range(count):
                    self._add(label=label, split=split, duration=2.0)
        for label in self.heldout_labels:
            for _ in range(4):
                self._add(label=label, split="eval", duration=2.0)
        for label in self.nuisance_labels:
            for split, count in (("train", 2), ("validation", 1), ("eval", 3)):
                for _ in range(count):
                    self._add(label=label, split=split, duration=6.0)

    def _write_all(self) -> None:
        _write_jsonl(self.fuss_manifest, self.fuss_rows)
        _write_jsonl(self.fsd_manifest, self.fsd_rows)
        self.fuss_license.write_text(
            "Creative Commons Attribution 4.0 International\n", encoding="utf-8"
        )
        self.fsd_license.write_text(
            "Creative Commons Attribution 4.0 International\n", encoding="utf-8"
        )
        self.refresh_pins()
        _write_json(
            self.selection_path,
            {
                "schema_version": SELECTION_FORMAT,
                "seen_labels": self.seen_labels,
                "heldout_labels": self.heldout_labels,
                "nuisance_labels": self.nuisance_labels,
            },
        )

    def refresh_fuss(self) -> None:
        _write_jsonl(self.fuss_manifest, self.fuss_rows)
        self.refresh_pins()

    def refresh_fsd(self) -> None:
        _write_jsonl(self.fsd_manifest, self.fsd_rows)
        self.refresh_pins()

    def refresh_pins(self) -> None:
        _write_json(
            self.pins_path,
            {
                "schema_version": PINS_FORMAT,
                "audit_date": "2026-07-21",
                "fuss": {
                    "dataset": "FUSS",
                    "dataset_version": "1.3",
                    "doi": "10.5281/zenodo.4012661",
                    "manifest_path": "metadata/fuss_sources.jsonl",
                    "manifest_sha256": sha256_file(self.fuss_manifest),
                    "license_path": "LICENSE",
                    "license_sha256": sha256_file(self.fuss_license),
                    "dataset_license_spdx": "CC-BY-4.0",
                    "dataset_license_url": (
                        "https://creativecommons.org/licenses/by/4.0/"
                    ),
                    "recipe_repository": (
                        "https://github.com/google-research/sound-separation"
                    ),
                    "recipe_revision": "a" * 40,
                },
                "fsd50k": {
                    "dataset": "FSD50K",
                    "dataset_version": "1.0",
                    "doi": "10.5281/zenodo.4060432",
                    "manifest_path": "metadata/fsd50k_labels_provenance.jsonl",
                    "manifest_sha256": sha256_file(self.fsd_manifest),
                    "license_path": "LICENSE",
                    "license_sha256": sha256_file(self.fsd_license),
                    "dataset_license_spdx": "CC-BY-4.0",
                    "dataset_license_url": (
                        "https://creativecommons.org/licenses/by/4.0/"
                    ),
                },
            },
        )

    def audit(self, *, verify_audio: bool = False):
        return audit_source_ledger(
            fuss_root=self.fuss_root,
            fsd50k_root=self.fsd50k_root,
            pins_path=self.pins_path,
            selection_path=self.selection_path,
            project_root=self.project_root,
            seed=314_159,
            verify_audio=verify_audio,
        )


class QCESV5SourceLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SyntheticLedger(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_metadata_plan_passes_without_claiming_audio_ready(self) -> None:
        plan, compliance, receipt = self.fixture.audit()
        self.assertEqual(plan["format"], PLAN_FORMAT)
        self.assertEqual(compliance["format"], COMPLIANCE_FORMAT)
        self.assertEqual(plan["source_count"], 388)
        self.assertEqual(plan["source_counts_by_partition"], EXPECTED_PARTITION_COUNTS)
        self.assertTrue(plan["metadata_gate_passed"])
        self.assertFalse(plan["acquisition_complete"])
        self.assertFalse(plan["release_ready"])
        self.assertFalse(compliance["release_ready"])
        self.assertIsNone(receipt)
        self.assertEqual(plan["metadata_path"], "qces_v5_source_pins.json")
        self.assertEqual(plan["metadata_sha256"], sha256_file(self.fixture.pins_path))
        self.assertTrue(all(not row["acquired"] for row in plan["sources"]))
        self.assertTrue(
            all(
                metric["direction"] in {"↑", "↓"}
                for metric in compliance["metrics"].values()
            )
        )

    def test_finalize_emits_exact_audited_receipt_after_hash_verification(self) -> None:
        plan, compliance, receipt = self.fixture.audit(verify_audio=True)
        self.assertTrue(plan["metadata_gate_passed"])
        self.assertTrue(compliance["release_ready"])
        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual(receipt["format"], RECEIPT_FORMAT)
        self.assertTrue(receipt["acquisition_complete"])
        self.assertTrue(receipt["release_ready"])
        self.assertEqual(receipt["source_count"], 388)
        self.assertTrue(all(row["acquired"] for row in receipt["sources"]))
        self.assertEqual(
            receipt["audio_verification"]["verified_source_count"], 388
        )

    def test_ambiguous_unmatched_and_missing_rows_are_rejected(self) -> None:
        self.fixture.fsd_rows[0]["labels"] = ["seen_00", "another_label"]
        self.fixture.fsd_rows.pop(1)
        del self.fixture.fsd_rows[1]["creator_id"]
        extra = dict(self.fixture.fsd_rows[2])
        extra.update(
            {
                "source_id": "fsd-only-source",
                "creator_id": fsd50k_uploader_key("fsd-only-uploader"),
                "uploader_id": fsd50k_uploader_key("fsd-only-uploader"),
                "uploader_name": "fsd-only-uploader",
            }
        )
        self.fixture.fsd_rows.append(extra)
        self.fixture.refresh_fsd()
        plan, compliance, receipt = self.fixture.audit()
        self.assertFalse(plan["metadata_gate_passed"])
        self.assertIsNone(receipt)
        metrics = compliance["metrics"]
        self.assertGreaterEqual(metrics["multilabel_ambiguous_count"]["value"], 1)
        self.assertGreaterEqual(metrics["unmatched_source_count"]["value"], 3)
        self.assertGreaterEqual(metrics["missing_required_field_count"]["value"], 1)
        self.assertGreater(metrics["allocation_shortfall_count"]["value"], 0)

    def test_missing_hash_and_license_fields_are_rejected(self) -> None:
        del self.fixture.fuss_rows[0]["sha256"]
        del self.fixture.fsd_rows[1]["source_license_spdx"]
        self.fixture.refresh_fuss()
        self.fixture.refresh_fsd()
        plan, compliance, _ = self.fixture.audit()
        self.assertFalse(plan["metadata_gate_passed"])
        self.assertGreaterEqual(
            compliance["metrics"]["missing_required_field_count"]["value"], 2
        )

    def test_uploader_cannot_cross_qces_partitions(self) -> None:
        train_row = next(
            row
            for row in self.fixture.fsd_rows
            if row["labels"] == ["seen_00"] and row["split"] == "train"
        )
        validation_row = next(
            row
            for row in self.fixture.fsd_rows
            if row["labels"] == ["seen_00"] and row["split"] == "validation"
        )
        validation_row["uploader_name"] = train_row["uploader_name"]
        validation_row["creator_id"] = train_row["creator_id"]
        validation_row["uploader_id"] = train_row["uploader_id"]
        self.fixture.refresh_fsd()
        plan, compliance, _ = self.fixture.audit()
        self.assertFalse(plan["metadata_gate_passed"])
        self.assertGreater(
            compliance["metrics"]["allocation_shortfall_count"]["value"], 0
        )
        self.assertEqual(
            compliance["metrics"]["uploader_split_leakage_count"]["value"], 0
        )

    def test_manifest_order_does_not_change_selected_sources(self) -> None:
        first_plan, _, _ = self.fixture.audit()
        self.fixture.fuss_rows.reverse()
        self.fixture.fsd_rows.reverse()
        self.fixture.refresh_fuss()
        self.fixture.refresh_fsd()
        second_plan, _, _ = self.fixture.audit()
        projection = lambda plan: [
            (row["source_id"], row["role"], row["partition"])
            for row in plan["sources"]
        ]
        self.assertEqual(projection(first_plan), projection(second_plan))

    def test_corrupt_pin_hash_is_a_hard_error(self) -> None:
        pins = json.loads(self.fixture.pins_path.read_text(encoding="utf-8"))
        pins["fuss"]["manifest_sha256"] = "0" * 64
        _write_json(self.fixture.pins_path, pins)
        with self.assertRaisesRegex(SourceAuditError, "manifest hash mismatch"):
            self.fixture.audit()

    def test_audio_hash_mismatch_blocks_final_receipt(self) -> None:
        source = self.fixture.fuss_rows[0]
        (self.fixture.fuss_root / source["audio_path"]).write_bytes(b"corrupted")
        plan, compliance, receipt = self.fixture.audit(verify_audio=True)
        self.assertTrue(plan["metadata_gate_passed"])
        self.assertFalse(compliance["release_ready"])
        self.assertIsNone(receipt)
        self.assertEqual(
            compliance["metrics"]["audio_missing_or_hash_mismatch_count"]["value"],
            1,
        )

    def test_cli_metadata_overwrite_invalidates_an_old_receipt(self) -> None:
        output_dir = self.fixture.project_root / "audit-output"
        common = [
            "--fuss-root",
            str(self.fixture.fuss_root),
            "--fsd50k-root",
            str(self.fixture.fsd50k_root),
            "--pins-json",
            str(self.fixture.pins_path),
            "--selection-json",
            str(self.fixture.selection_path),
            "--project-root",
            str(self.fixture.project_root),
            "--output-dir",
            str(output_dir),
        ]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(audit_cli_main(common + ["--finalize"]), 0)
            self.assertEqual(audit_cli_main(common + ["--overwrite"]), 0)
        tombstone = json.loads(
            (output_dir / RECEIPT_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(tombstone["format"], BLOCKED_RECEIPT_FORMAT)
        self.assertFalse(tombstone["acquisition_complete"])


if __name__ == "__main__":
    unittest.main()
