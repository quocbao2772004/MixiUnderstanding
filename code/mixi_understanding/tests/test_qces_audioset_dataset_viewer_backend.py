from __future__ import annotations

import io
import json
import unittest

import numpy as np
import soundfile as sf

from mixi_understanding.qces.audioset_dataset_viewer_backend import (
    BINDING_FORMAT,
    DatasetViewerTransportError,
    ViewerRetryPolicy,
    build_split_binding,
    fetch_row_audio,
    resolve_row_location,
)


REVISION = "a" * 40
LFS_A = "1" * 64
XET_A = "2" * 64
LFS_B = "3" * 64
XET_B = "4" * 64


def _flac_bytes() -> bytes:
    sample_rate = 16_000
    times = np.arange(2_000, dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * times)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="FLAC")
    return output.getvalue()


def _source(
    name: str,
    *,
    size: int,
    lfs: str,
    xet: str,
    groups: list[int],
) -> dict[str, object]:
    return {
        "source_route": "route-v1",
        "shard_id": f"shard-{name}",
        "hf_dataset": "org/dataset",
        "hf_revision": REVISION,
        "hf_split": "train",
        "parquet_url": f"hf://datasets/org/dataset@{REVISION}/data/{name}.parquet",
        "size_bytes": size,
        "lfs_sha256": lfs,
        "xet_hash": xet,
        "num_rows": sum(groups),
        "row_groups": [
            {"index": index, "num_rows": count}
            for index, count in enumerate(groups)
        ],
    }


def _binding() -> dict[str, object]:
    parquet_files = [
        {
            "dataset": "org/dataset",
            "config": "full",
            "split": "train-view",
            "url": (
                "https://huggingface.co/datasets/org/dataset/resolve/"
                "refs%2Fconvert%2Fparquet/full/train-view/0001.parquet"
            ),
            "size": 20,
        },
        {
            "dataset": "org/dataset",
            "config": "full",
            "split": "train-view",
            "url": (
                "https://huggingface.co/datasets/org/dataset/resolve/"
                "refs%2Fconvert%2Fparquet/full/train-view/0000.parquet"
            ),
            "size": 10,
        },
    ]
    # The official order is deliberately B, A.  It must not be replaced by a
    # lexical sort of either source paths or export filenames.
    return build_split_binding(
        dataset="org/dataset",
        source_revision=REVISION,
        config="full",
        viewer_split="train-view",
        hf_split="train",
        source_shards=[
            _source("source-a", size=10, lfs=LFS_A, xet=XET_A, groups=[2, 3]),
            _source("source-b", size=20, lfs=LFS_B, xet=XET_B, groups=[4]),
        ],
        parquet_document={
            "partial": False,
            "pending": [],
            "failed": [],
            "parquet_files": parquet_files,
        },
        parquet_response_revision=REVISION,
        parquet_response_sha256="5" * 64,
        convert_commit="b" * 40,
        convert_tree_entries=[
            {
                "type": "file",
                "path": "full/train-view/0000.parquet",
                "size": 10,
                "lfs": {"oid": LFS_A, "size": 10},
                "xetHash": XET_A,
            },
            {
                "type": "file",
                "path": "full/train-view/0001.parquet",
                "size": 20,
                "lfs": {"oid": LFS_B, "size": 20},
                "xetHash": XET_B,
            },
        ],
    )


def _availability(*, source: str = "source-a", row_group: int = 1, row: int = 2):
    if source == "source-a":
        size, lfs, xet = 10, LFS_A, XET_A
    else:
        size, lfs, xet = 20, LFS_B, XET_B
    return {
        "source_route": "route-v1",
        "hf_dataset": "org/dataset",
        "hf_revision": REVISION,
        "hf_split": "train",
        "video_id": "video-7",
        "labels": ["/m/one", "/m/two"],
        "human_labels": ["One", "Two"],
        "parquet_url": f"hf://datasets/org/dataset@{REVISION}/data/{source}.parquet",
        "row_group": row_group,
        "row_index": row,
        "shard_provenance": {
            "file_size_bytes": size,
            "lfs_sha256": lfs,
            "xet_hash": xet,
        },
    }


class _Response:
    def __init__(self, status: int, content: bytes, *, headers=None, document=None):
        self.status_code = status
        self.content = content
        self.headers = dict(headers or {})
        self._document = document

    def json(self):
        if self._document is not None:
            return self._document
        return json.loads(self.content)


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected request")
        return self.responses.pop(0)


def _success_session(location, *, video_id="video-7", revision=REVISION):
    audio = _flac_bytes()
    asset_url = (
        "https://datasets-server.huggingface.co/cached-assets/org/dataset/--/"
        f"{REVISION}/--/full/train-view/{location['global_row']}/audio/audio.flac?"
        "Expires=1&Signature=sig&Key-Pair-Id=key"
    )
    row_document = {
        "partial": False,
        "num_rows_total": 100,
        "rows": [
            {
                "row_idx": location["global_row"],
                "row": {
                    "video_id": video_id,
                    "labels": ["/m/one", "/m/two"],
                    "human_labels": ["One", "Two"],
                    "audio": [{"src": asset_url, "type": "audio/flac"}],
                },
                "truncated_cells": [],
            }
        ],
    }
    row_bytes = json.dumps(row_document).encode()
    return _Session(
        [
            _Response(
                200,
                row_bytes,
                headers={"x-revision": revision},
                document=row_document,
            ),
            _Response(
                200,
                audio,
                headers={"Content-Length": str(len(audio))},
            ),
        ]
    ), audio, len(row_bytes)


class DatasetViewerBackendTests(unittest.TestCase):
    def test_binding_retains_official_file_order_and_resolves_global_row(self):
        binding = _binding()
        self.assertEqual(binding["format"], BINDING_FORMAT)
        self.assertEqual(binding["files"][0]["source_parquet_url"].split("/")[-1], "source-b.parquet")
        self.assertEqual(binding["files"][1]["source_parquet_url"].split("/")[-1], "source-a.parquet")
        # Source B contributes four rows first; source A row-group 0 adds two.
        location = resolve_row_location(binding, _availability())
        self.assertEqual(location["global_row"], 4 + 2 + 2)
        self.assertEqual(location["viewer_file_ordinal"], 1)

    def test_binding_rejects_non_identical_convert_export(self):
        binding = _binding()
        source_shards = []
        for row in binding["files"]:
            source_shards.append(
                _source(
                    row["source_parquet_url"].rsplit("/", 1)[-1].split(".")[0],
                    size=row["source_size_bytes"],
                    lfs=row["source_lfs_sha256"],
                    xet=row["source_xet_hash"],
                    groups=row["source_row_group_rows"],
                )
            )
        with self.assertRaisesRegex(
            DatasetViewerTransportError, "content-identical"
        ):
            build_split_binding(
                dataset="org/dataset",
                source_revision=REVISION,
                config="full",
                viewer_split="train-view",
                hf_split="train",
                source_shards=source_shards,
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
                            "size": 10,
                        },
                        {
                            "dataset": "org/dataset",
                            "config": "full",
                            "split": "train-view",
                            "url": "https://huggingface.co/datasets/org/dataset/resolve/refs%2Fconvert%2Fparquet/full/train-view/0001.parquet",
                            "size": 20,
                        },
                    ],
                },
                parquet_response_revision=REVISION,
                parquet_response_sha256="5" * 64,
                convert_commit="b" * 40,
                convert_tree_entries=[
                    {
                        "type": "file",
                        "path": "full/train-view/0000.parquet",
                        "size": 10,
                        "lfs": {"oid": "6" * 64, "size": 10},
                        "xetHash": XET_A,
                    },
                    {
                        "type": "file",
                        "path": "full/train-view/0001.parquet",
                        "size": 20,
                        "lfs": {"oid": LFS_B, "size": 20},
                        "xetHash": XET_B,
                    },
                ],
            )

    def test_location_rejects_shard_provenance_change(self):
        availability = _availability()
        availability["shard_provenance"]["xet_hash"] = "f" * 64
        with self.assertRaisesRegex(
            DatasetViewerTransportError, "provenance mismatch"
        ):
            resolve_row_location(_binding(), availability)

    def test_fetch_validates_audio_and_counts_response_payload(self):
        availability = _availability()
        availability["labels"] = ["/m/two", "/m/one"]
        availability["human_labels"] = ["Two", "One"]
        location = resolve_row_location(_binding(), availability)
        session, expected_audio, row_bytes = _success_session(location)
        actual_audio, receipt = fetch_row_audio(
            location,
            session=session,
            retry=ViewerRetryPolicy(max_attempts=1),
        )
        self.assertEqual(actual_audio, expected_audio)
        self.assertEqual(receipt["rows_response_payload_bytes"], row_bytes)
        self.assertEqual(receipt["asset_response_payload_bytes"], len(expected_audio))
        self.assertEqual(
            receipt["total_response_payload_bytes"], row_bytes + len(expected_audio)
        )
        self.assertEqual(receipt["sample_rate"], 16_000)
        self.assertEqual(receipt["frames"], 2_000)
        self.assertEqual(len(receipt["decoded_pcm_f32le_sha256"]), 64)
        self.assertNotIn("Signature=", json.dumps(receipt))

    def test_fetch_rejects_revision_mismatch(self):
        location = resolve_row_location(_binding(), _availability())
        session, _, _ = _success_session(location, revision="c" * 40)
        with self.assertRaisesRegex(
            DatasetViewerTransportError, "x-revision"
        ):
            fetch_row_audio(
                location,
                session=session,
                retry=ViewerRetryPolicy(max_attempts=1),
            )
        self.assertEqual(len(session.calls), 1)

    def test_fetch_rejects_video_id_mismatch_before_asset_download(self):
        location = resolve_row_location(_binding(), _availability())
        session, _, _ = _success_session(location, video_id="wrong")
        with self.assertRaisesRegex(
            DatasetViewerTransportError, "video_id mismatch"
        ):
            fetch_row_audio(
                location,
                session=session,
                retry=ViewerRetryPolicy(max_attempts=1),
            )
        self.assertEqual(len(session.calls), 1)

    def test_fetch_retries_transient_rows_error_without_fallback(self):
        location = resolve_row_location(_binding(), _availability())
        good_session, expected_audio, _ = _success_session(location)
        transient = _Response(429, b"queue full")
        session = _Session([transient, *good_session.responses])
        _, receipt = fetch_row_audio(
            location,
            session=session,
            retry=ViewerRetryPolicy(
                max_attempts=2,
                initial_delay_seconds=0,
                maximum_delay_seconds=0,
            ),
            sleep=lambda _: None,
        )
        self.assertEqual(receipt["rows_attempts"], 2)
        self.assertGreater(receipt["rows_response_payload_bytes"], len(b"queue full"))
        self.assertEqual(
            receipt["total_response_payload_bytes"],
            receipt["rows_response_payload_bytes"] + len(expected_audio),
        )


if __name__ == "__main__":
    unittest.main()
