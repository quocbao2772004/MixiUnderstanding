"""Tests for the official FUSS/FSD50K-to-QCES v5 normalizer."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any, Dict, List

from mixi_understanding.data.qces_v5_source_ledger import (
    FSD50KSource,
    SourceAuditError,
    fsd50k_uploader_key,
)
from mixi_understanding.data.qces_v5_upstream_normalizer import (
    FUSS_MANIFEST_NAME,
    REPORT_FORMAT,
    normalize_official_upstream,
    write_normalized_outputs,
)


def _write_csv(path: Path, fields: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_wav(path: Path, *, frames: int = 32_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * frames)


def _zero_riff_size(path: Path) -> None:
    """Reproduce the RIFF-size anomaly present in official FUSS v1.3."""

    with path.open("r+b") as handle:
        if handle.read(4) != b"RIFF":
            raise AssertionError("fixture is not RIFF")
        handle.write(b"\0\0\0\0")


class SyntheticOfficialUpstream:
    def __init__(self, root: Path) -> None:
        self.fuss_root = root / "fuss_v1.3"
        self.fuss_data_dir = self.fuss_root / "fsd_data"
        self.ground_truth_dir = root / "FSD50K.ground_truth"
        self.metadata_dir = root / "FSD50K.metadata"
        self.records = [
            ("1001", "train", "foreground", "Leaf_train_fg", "native-val"),
            ("1002", "train", "background", "Leaf_train_bg", "native-train"),
            (
                "1003",
                "validation",
                "foreground",
                "Leaf_validation_fg",
                "native-train",
            ),
            (
                "1004",
                "validation",
                "background",
                "Leaf_validation_bg",
                "native-val",
            ),
            ("1005", "eval", "foreground", "Leaf_eval_fg", "eval"),
            ("1006", "eval", "background", "Leaf_eval_bg", "eval"),
        ]
        self.clip_info: Dict[str, Dict[str, Dict[str, Any]]] = {
            "dev": {},
            "eval": {},
        }
        self.collection_rows: Dict[str, List[Dict[str, str]]] = {
            "dev": [],
            "eval": [],
        }
        self.ground_truth_rows: Dict[str, List[Dict[str, str]]] = {
            "dev": [],
            "eval": [],
        }
        self._write_inputs()

    def _write_inputs(self) -> None:
        list_rows: Dict[tuple[str, str], List[str]] = {}
        for source_id, split, list_kind, label, native_split in self.records:
            list_rows.setdefault((split, list_kind), []).append(
                f"{split}/sound/{source_id}.wav"
            )
            _write_wav(self.fuss_data_dir / split / "sound" / f"{source_id}.wav")
            partition = "eval" if split == "eval" else "dev"
            ground_truth = {
                "fname": source_id,
                "labels": f"{label},Parent",
                "mids": "/m/leaf,/m/parent",
            }
            if partition == "dev":
                ground_truth["split"] = native_split
            self.ground_truth_rows[partition].append(ground_truth)
            self.collection_rows[partition].append(
                {"fname": source_id, "labels": label, "mids": "/m/leaf"}
            )
            self.clip_info[partition][source_id] = {
                "title": f"Title {source_id}",
                "description": "synthetic official-shape fixture",
                "tags": ["fixture"],
                "license": (
                    "http://creativecommons.org/publicdomain/zero/1.0/"
                ),
                "uploader": f"Uploader_{source_id}",
            }
        for (split, list_kind), rows in list_rows.items():
            path = self.fuss_data_dir / f"{split}_{list_kind}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        _write_csv(
            self.ground_truth_dir / "dev.csv",
            ["fname", "labels", "mids", "split"],
            self.ground_truth_rows["dev"],
        )
        _write_csv(
            self.ground_truth_dir / "eval.csv",
            ["fname", "labels", "mids"],
            self.ground_truth_rows["eval"],
        )
        _write_csv(
            self.metadata_dir / "collection" / "collection_dev.csv",
            ["fname", "labels", "mids"],
            self.collection_rows["dev"],
        )
        _write_csv(
            self.metadata_dir / "collection" / "collection_eval.csv",
            ["fname", "labels", "mids"],
            self.collection_rows["eval"],
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        for partition in ("dev", "eval"):
            (self.metadata_dir / f"{partition}_clips_info_FSD50K.json").write_text(
                json.dumps(self.clip_info[partition], sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def refresh_collection(self) -> None:
        for partition in ("dev", "eval"):
            _write_csv(
                self.metadata_dir / "collection" / f"collection_{partition}.csv",
                ["fname", "labels", "mids"],
                self.collection_rows[partition],
            )

    def refresh_clips(self) -> None:
        for partition in ("dev", "eval"):
            (self.metadata_dir / f"{partition}_clips_info_FSD50K.json").write_text(
                json.dumps(self.clip_info[partition], sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def normalize(self, *, metadata_only: bool = False):
        return normalize_official_upstream(
            fuss_root=self.fuss_root,
            fuss_data_dir=self.fuss_data_dir,
            fsd50k_ground_truth_dir=self.ground_truth_dir,
            fsd50k_metadata_dir=self.metadata_dir,
            metadata_only=metadata_only,
        )


class QCESV5UpstreamNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SyntheticOfficialUpstream(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_normalization_uses_collection_label_and_fuss_split(self) -> None:
        result = self.fixture.normalize()
        self.assertEqual(result["report"]["format"], REPORT_FORMAT)
        self.assertEqual(len(result["fuss_rows"]), 6)
        self.assertEqual(len(result["fsd50k_rows"]), 6)
        self.assertEqual(result["rejections"], [])
        train = next(row for row in result["fsd50k_rows"] if row["source_id"] == "1001")
        self.assertEqual(train["split"], "train")
        self.assertEqual(train["labels"], ["Leaf_train_fg"])
        expected_key = fsd50k_uploader_key("Uploader_1001")
        self.assertEqual(train["creator_id"], expected_key)
        self.assertEqual(train["uploader_id"], expected_key)
        self.assertEqual(train["uploader_name"], "Uploader_1001")
        self.assertFalse(result["report"]["identity_key_scheme"]["upstream_numeric_id_claimed"])
        self.assertTrue(
            all(
                metric["direction"] in {"↑", "↓"}
                for metric in result["report"]["metrics"].values()
            )
        )
        FSD50KSource.from_dict(train, "normalized")

    def test_row_level_defects_are_rejected_without_aborting(self) -> None:
        self.fixture.collection_rows["dev"].append(
            {"fname": "1001", "labels": "Second_leaf", "mids": "/m/second"}
        )
        self.fixture.clip_info["dev"]["1002"]["license"] = (
            "http://creativecommons.org/licenses/by/3.0/"
        )
        with (self.fixture.fuss_data_dir / "validation_foreground.txt").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("validation/sound/1999.wav\n")
        _write_wav(
            self.fixture.fuss_data_dir / "validation" / "sound" / "1999.wav"
        )
        self.fixture.refresh_collection()
        self.fixture.refresh_clips()
        result = self.fixture.normalize()
        reasons = {row["reason"] for row in result["rejections"]}
        self.assertEqual(
            reasons,
            {
                "ambiguous_collection_label",
                "source_license_not_cc0",
                "unmatched_fsd50k_source_id",
            },
        )
        self.assertEqual(len(result["fsd50k_rows"]), 4)
        self.assertTrue(result["report"]["conversion_completed"])
        self.assertFalse(result["report"]["release_ready"])

    def test_metadata_only_accepts_join_without_audio_and_emits_no_fuss_manifest(self) -> None:
        missing_audio = self.fixture.fuss_data_dir / "train" / "sound" / "1001.wav"
        missing_audio.unlink()
        result = self.fixture.normalize(metadata_only=True)
        self.assertEqual(len(result["fsd50k_rows"]), 6)
        self.assertEqual(len(result["fuss_rows"]), 0)
        self.assertEqual(
            result["report"]["metrics"]["missing_audio_count"]["value"], 0
        )
        output_dir = Path(self.temporary.name) / "metadata-only-output"
        paths = write_normalized_outputs(result, output_dir=output_dir)
        self.assertNotIn("fuss_manifest", paths)
        self.assertFalse((output_dir / FUSS_MANIFEST_NAME).exists())

    def test_zero_sized_riff_header_is_verified_without_dropping_source(self) -> None:
        anomalous = self.fixture.fuss_data_dir / "train" / "sound" / "1001.wav"
        _zero_riff_size(anomalous)
        result = self.fixture.normalize()
        row = next(row for row in result["fuss_rows"] if row["source_id"] == "1001")
        self.assertEqual(row["duration_seconds"], 2.0)
        self.assertEqual(
            result["report"]["metrics"]["invalid_audio_count"]["value"], 0
        )
        self.assertEqual(
            result["report"]["metrics"][
                "official_zero_sized_riff_recovered_count"
            ]["value"],
            1,
        )

    def test_zero_sized_riff_fallback_rejects_truncated_chunks(self) -> None:
        malformed = self.fixture.fuss_data_dir / "train" / "sound" / "1001.wav"
        malformed.write_bytes(b"RIFF\0\0\0\0WAVEfmt \xff\xff\xff\xff")
        result = self.fixture.normalize()
        rejected = [
            row
            for row in result["rejections"]
            if row["source_id"] == "1001" and row["reason"] == "invalid_fuss_audio"
        ]
        self.assertEqual(len(rejected), 1)
        self.assertIn("exceeds the physical file size", rejected[0]["detail"])

    def test_ledger_recomputes_identity_key_from_exact_username(self) -> None:
        row = self.fixture.normalize()["fsd50k_rows"][0]
        row["uploader_id"] = fsd50k_uploader_key(row["uploader_name"].lower())
        with self.assertRaisesRegex(SourceAuditError, "deterministic key"):
            FSD50KSource.from_dict(row, "tampered")


class QCESV5OfficialMetadataSmokeTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    STAGING_ROOT = (
        PROJECT_ROOT / "upstream" / "_staging" / "qces_v5_metadata_20260722"
    )
    FUSS_DATA_DIR = (
        STAGING_ROOT
        / "fuss_v1.3"
        / "partial_archive_metadata"
        / "fsd_data"
    )
    FSD_EXTRACTED = STAGING_ROOT / "fsd50k_v1.0" / "extracted"

    @unittest.skipUnless(
        (FUSS_DATA_DIR / "validation_foreground.txt").is_file()
        and (FUSS_DATA_DIR / "validation_background.txt").is_file()
        and (FSD_EXTRACTED / "FSD50K.ground_truth" / "dev.csv").is_file(),
        "official metadata-only staging snapshot is not present",
    )
    def test_official_validation_lists_join_without_audio(self) -> None:
        result = normalize_official_upstream(
            fuss_root=self.FUSS_DATA_DIR.parent,
            fuss_data_dir=self.FUSS_DATA_DIR,
            fsd50k_ground_truth_dir=self.FSD_EXTRACTED / "FSD50K.ground_truth",
            fsd50k_metadata_dir=self.FSD_EXTRACTED / "FSD50K.metadata",
            splits=("validation",),
            metadata_only=True,
        )
        metrics = result["report"]["metrics"]
        self.assertEqual(metrics["fuss_listed_source_count"]["value"], 2883)
        self.assertEqual(metrics["normalized_fsd50k_row_count"]["value"], 2516)
        self.assertEqual(metrics["unmatched_fsd50k_source_count"]["value"], 13)
        self.assertEqual(metrics["ambiguous_collection_label_count"]["value"], 0)
        self.assertEqual(metrics["ground_truth_label_mismatch_count"]["value"], 354)
        self.assertEqual(metrics["non_cc0_source_count"]["value"], 0)
        self.assertTrue(
            all(
                row["creator_id"] == row["uploader_id"]
                and row["creator_id"]
                == fsd50k_uploader_key(row["uploader_name"])
                for row in result["fsd50k_rows"]
            )
        )


if __name__ == "__main__":
    unittest.main()
