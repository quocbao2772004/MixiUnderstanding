from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from mixi_understanding.qces.audioset_dataset_viewer_backend import (
    FETCH_FORMAT,
    DatasetViewerTransportError,
    _decoded_audio_receipt,
    build_split_binding,
)
from mixi_understanding.qces.audioset_dataset_viewer_materializer import (
    ViewerMaterializationConfig,
    run_viewer_materialization,
)
from mixi_understanding.qces.audioset_plan_materializer import (
    load_existing_audio_sources,
)


REVISION = "a" * 40
LFS = "1" * 64
XET = "2" * 64


def _flac_bytes() -> bytes:
    sample_rate = 16_000
    times = np.arange(1_600, dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 330.0 * times)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="FLAC")
    return output.getvalue()


def _binding() -> dict[str, object]:
    source = {
        "source_route": "route-v1",
        "shard_id": "shard-1",
        "hf_dataset": "org/dataset",
        "hf_revision": REVISION,
        "hf_split": "train",
        "parquet_url": f"hf://datasets/org/dataset@{REVISION}/data/a.parquet",
        "size_bytes": 100,
        "lfs_sha256": LFS,
        "xet_hash": XET,
        "num_rows": 2,
        "row_groups": [{"index": 0, "num_rows": 2}],
    }
    return build_split_binding(
        dataset="org/dataset",
        source_revision=REVISION,
        config="full",
        viewer_split="train-view",
        hf_split="train",
        source_shards=[source],
        parquet_document={
            "partial": False,
            "pending": [],
            "failed": [],
            "parquet_files": [
                {
                    "dataset": "org/dataset",
                    "config": "full",
                    "split": "train-view",
                    "url": "https://huggingface.co/datasets/org/dataset/resolve/refs%2Fconvert%2Fparquet/full/train-view/0000.parquet",
                    "size": 100,
                }
            ],
        },
        parquet_response_revision=REVISION,
        parquet_response_sha256="3" * 64,
        convert_commit="b" * 40,
        convert_tree_entries=[
            {
                "type": "file",
                "path": "full/train-view/0000.parquet",
                "size": 100,
                "lfs": {"oid": LFS, "size": 100},
                "xetHash": XET,
            }
        ],
    )


def _availability() -> dict[str, object]:
    return {
        "source_route": "route-v1",
        "hf_dataset": "org/dataset",
        "hf_revision": REVISION,
        "hf_split": "train",
        "video_id": "video_1",
        "labels": ["/m/test"],
        "human_labels": ["Test"],
        "parquet_url": f"hf://datasets/org/dataset@{REVISION}/data/a.parquet",
        "row_group": 0,
        "row_index": 1,
        "shard_provenance": {
            "file_size_bytes": 100,
            "lfs_sha256": LFS,
            "xet_hash": XET,
        },
    }


class DatasetViewerMaterializerTests(unittest.TestCase):
    def test_transaction_resume_and_existing_manifest_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding_path = root / "binding.json"
            binding_path.write_text(json.dumps(_binding()), encoding="utf-8")
            availability_path = root / "availability.json"
            availability_path.write_text(json.dumps(_availability()), encoding="utf-8")
            calls = []

            def fetcher(location, **_):
                calls.append(location["video_id"])
                audio = _flac_bytes()
                receipt = {
                    "format": FETCH_FORMAT,
                    **_decoded_audio_receipt(audio),
                    "total_response_payload_bytes": len(audio) + 123,
                    "receipt_sha256": "4" * 64,
                }
                return audio, receipt

            config = ViewerMaterializationConfig(
                binding_path=binding_path,
                availability_paths=(availability_path,),
                output_dir=root / "output",
                minimum_free_disk_bytes=0,
            )
            first = run_viewer_materialization(config, fetcher=fetcher)
            second = run_viewer_materialization(config, fetcher=fetcher)
            self.assertTrue(first["complete"])
            self.assertEqual(first["new_rows_this_run"], 1)
            self.assertEqual(second["new_rows_this_run"], 0)
            self.assertEqual(calls, ["video_1"])

            manifest = Path(first["source_audio_manifest"])
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(rows[0]["transport"], "hf_dataset_viewer_row")
            sources, receipts = load_existing_audio_sources(
                [manifest], plans={"video_1": {"hf_split": "train"}}
            )
            self.assertIn("video_1", sources)
            self.assertEqual(receipts[0]["matched_plan_rows"], 1)

    def test_corrupt_committed_audio_fails_closed_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding_path = root / "binding.json"
            binding_path.write_text(json.dumps(_binding()), encoding="utf-8")
            availability_path = root / "availability.json"
            availability_path.write_text(json.dumps(_availability()), encoding="utf-8")

            def fetcher(location, **_):
                audio = _flac_bytes()
                return audio, {
                    "format": FETCH_FORMAT,
                    **_decoded_audio_receipt(audio),
                    "total_response_payload_bytes": len(audio),
                    "receipt_sha256": "4" * 64,
                }

            config = ViewerMaterializationConfig(
                binding_path=binding_path,
                availability_paths=(availability_path,),
                output_dir=root / "output",
                minimum_free_disk_bytes=0,
            )
            run_viewer_materialization(config, fetcher=fetcher)
            audio_path = next((root / "output/transactions").glob("*/audio.flac"))
            audio_path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(
                DatasetViewerTransportError, "audio hash mismatch"
            ):
                run_viewer_materialization(config, fetcher=fetcher)


if __name__ == "__main__":
    unittest.main()

