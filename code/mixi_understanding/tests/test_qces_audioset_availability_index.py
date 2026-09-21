from __future__ import annotations

import hashlib
import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq

from mixi_understanding.qces.audioset_availability_index import (
    AvailabilityAllowlist,
    AvailabilityIndexConfig,
    AvailabilityIndexError,
    AvailabilityRoute,
    RetryPolicy,
    run_availability_index,
)
from mixi_understanding.qces import audioset_availability_index as availability_module
from mixi_understanding.scripts.build_qces_audioset_availability_index import (
    main as availability_main,
)


def _write_shard(
    path: Path,
    rows: list[tuple[str, list[str], list[str]]],
    *,
    include_human_labels: bool = True,
    row_group_size: int = 2,
) -> None:
    values = []
    for video_id, labels, human_labels in rows:
        row = {"video_id": video_id, "labels": labels}
        if include_human_labels:
            row["human_labels"] = human_labels
        values.append(row)
    fields = [
        pa.field("video_id", pa.string()),
        pa.field("labels", pa.list_(pa.string())),
    ]
    if include_human_labels:
        fields.append(pa.field("human_labels", pa.list_(pa.string())))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(values, schema=pa.schema(fields)),
        path,
        row_group_size=row_group_size,
    )


def _allowlist(
    video_ids: set[str],
    *,
    source_route: str = "*",
    hf_split: str = "*",
) -> AvailabilityAllowlist:
    normalized = hashlib.sha256(
        json.dumps(
            sorted(video_ids),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return AvailabilityAllowlist(
        video_ids=frozenset(video_ids),
        normalized_sha256=normalized,
        source_path="synthetic-allowlist",
        source_sha256="a" * 64,
        input_rows=len(video_ids),
        source_route=source_route,
        hf_split=hf_split,
    )


def _route(
    root: Path,
    *,
    name: str,
    splits: tuple[str, ...],
) -> AvailabilityRoute:
    return AvailabilityRoute(
        source_route=name,
        hf_dataset=f"local/{name}",
        resolved_revision=f"revision-{name}",
        parquet_pattern=str(root / name / "{split}-*.parquet"),
        splits=splits,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class AudioSetAvailabilityIndexTests(unittest.TestCase):
    def test_remote_reader_disables_nested_readahead_cache(self) -> None:
        remote_fs = Mock()
        remote_fs.protocol = "https"
        sentinel = object()
        remote_fs.open.return_value = sentinel

        result = availability_module._open_for_parquet(
            remote_fs,
            "dataset/shard.parquet",
            1 << 20,
        )

        self.assertIs(result, sentinel)
        remote_fs.open.assert_called_once_with(
            "dataset/shard.parquet",
            "rb",
            block_size=1 << 20,
            cache_type="none",
        )

    def test_local_reader_does_not_pass_remote_cache_options(self) -> None:
        local_fs = Mock()
        local_fs.protocol = ("file", "local")
        sentinel = object()
        local_fs.open.return_value = sentinel

        result = availability_module._open_for_parquet(
            local_fs,
            "/tmp/shard.parquet",
            1 << 20,
        )

        self.assertIs(result, sentinel)
        local_fs.open.assert_called_once_with("/tmp/shard.parquet", "rb")

    def test_hf_route_refuses_unpinned_pattern(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not embed"):
            AvailabilityRoute(
                source_route="unsafe",
                hf_dataset="owner/dataset",
                resolved_revision="a" * 40,
                parquet_pattern="hf://datasets/owner/dataset/data/{split}-*.parquet",
                splits=("train",),
            )

    def test_output_lock_fails_closed_before_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_shard(
                root / "route/train-000.parquet",
                [("a", ["/a"], ["A"])],
            )
            output_dir = root / "output"
            output_dir.mkdir()
            config = AvailabilityIndexConfig(
                routes=(_route(root, name="route", splits=("train",)),),
                output_dir=output_dir,
                retry=RetryPolicy(max_attempts=1),
            )
            with (output_dir / ".availability_index.lock").open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(
                    AvailabilityIndexError, "holds the output lock"
                ):
                    run_availability_index(config)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def test_multiroute_allowlist_scans_all_rows_and_retains_duplicate_locations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_shard(
                root / "route_a/train-000.parquet",
                [
                    ("a", ["/m/z", "/m/a"], ["Zed", "Alpha"]),
                    ("b", ["/m/b"], ["Beta"]),
                    ("c", ["/m/c"], ["Gamma"]),
                ],
            )
            _write_shard(
                root / "route_a/test-000.parquet",
                [("d", ["/m/d"], ["Delta"]), ("e", ["/m/e"], ["Echo"])],
            )
            _write_shard(
                root / "route_b/bal_train-000.parquet",
                [("x", ["/m/x"], []), ("a", ["/m/a"], [])],
                include_human_labels=False,
            )
            routes = (
                _route(root, name="route_a", splits=("train", "test")),
                _route(root, name="route_b", splits=("bal_train",)),
            )
            common = dict(
                routes=routes,
                output_dir=root / "output",
                allowlists=(_allowlist({"a", "d", "x", "missing"}),),
            )
            first = run_availability_index(
                AvailabilityIndexConfig(
                    **common,
                    max_new_row_groups=2,
                    retry=RetryPolicy(max_attempts=1),
                )
            )
            self.assertFalse(first["complete"])
            self.assertLess(first["mirror_rows_scanned"], 7)

            receipt = run_availability_index(
                AvailabilityIndexConfig(
                    **common,
                    retry=RetryPolicy(max_attempts=1),
                )
            )
            self.assertTrue(receipt["complete"])
            self.assertTrue(receipt["integrity_pass"])
            self.assertEqual(receipt["mirror_rows_total"], 7)
            self.assertEqual(receipt["mirror_rows_scanned"], 7)
            self.assertEqual(receipt["indexed_allowlist_matches"], 4)
            self.assertEqual(
                receipt["allowlist_audit"]["unmatched_scope_video_ids"], 1
            )
            self.assertEqual(
                receipt["duplicates"]["video_ids_with_multiple_locations"], 1
            )
            self.assertEqual(receipt["duplicates"]["cross_route_duplicate_video_ids"], 1)
            self.assertEqual(
                receipt["duplicates"]["within_route_duplicate_video_ids"], 0
            )
            self.assertFalse(receipt["projection"]["audio_column_read"])

            rows = _read_jsonl(root / "output/audioset_availability_index.jsonl")
            self.assertEqual([row["video_id"] for row in rows].count("a"), 2)
            self.assertEqual({row["video_id"] for row in rows}, {"a", "d", "x"})
            route_a_a = next(
                row
                for row in rows
                if row["video_id"] == "a" and row["source_route"] == "route_a"
            )
            self.assertEqual(route_a_a["labels"], ["/m/a", "/m/z"])
            self.assertEqual(route_a_a["human_labels"], ["Alpha", "Zed"])
            route_b_x = next(row for row in rows if row["video_id"] == "x")
            self.assertEqual(route_b_x["human_labels"], [])
            self.assertIn("shard_metadata_sha256", route_b_x["shard_provenance"])

    def test_fragment_before_state_interruption_resumes_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_shard(
                root / "route/train-000.parquet",
                [
                    ("a", ["/a"], ["A"]),
                    ("b", ["/b"], ["B"]),
                    ("c", ["/c"], ["C"]),
                ],
                row_group_size=1,
            )
            config = AvailabilityIndexConfig(
                routes=(_route(root, name="route", splits=("train",)),),
                output_dir=root / "output",
                allowlists=(_allowlist({"a", "b", "c"}),),
                retry=RetryPolicy(max_attempts=1),
            )
            interrupted = False

            def hook(event: str, payload: object) -> None:
                nonlocal interrupted
                if event == "row_group_fragment_written" and not interrupted:
                    interrupted = True
                    raise RuntimeError("simulated interruption before state commit")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_availability_index(config, checkpoint_hook=hook)
            self.assertEqual(
                len(list((root / "output/row_group_fragments").glob("*.json"))), 1
            )
            receipt = run_availability_index(config)
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["indexed_allowlist_matches"], 3)

            fragment_path = sorted(
                (root / "output/row_group_fragments").glob("*.json")
            )[0]
            fragment = json.loads(fragment_path.read_text())
            fragment["entries"][0]["video_id"] = "tampered"
            fragment_path.write_text(json.dumps(fragment), encoding="utf-8")
            with self.assertRaisesRegex(
                AvailabilityIndexError, "fragment payload SHA-256 mismatch"
            ):
                run_availability_index(config)
            fragment.pop("fragment_payload_sha256")
            fragment["fragment_payload_sha256"] = hashlib.sha256(
                json.dumps(
                    fragment,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            fragment_path.write_text(json.dumps(fragment), encoding="utf-8")
            with self.assertRaisesRegex(
                AvailabilityIndexError, "outside effective allowlist"
            ):
                run_availability_index(config)

    def test_strict_duplicate_policy_writes_failed_integrity_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_shard(
                root / "route/train-000.parquet",
                [("duplicate", ["/a"], ["A"])],
            )
            _write_shard(
                root / "route/train-001.parquet",
                [("duplicate", ["/b"], ["B"])],
            )
            config = AvailabilityIndexConfig(
                routes=(_route(root, name="route", splits=("train",)),),
                output_dir=root / "output",
                require_unique_video_ids=True,
                retry=RetryPolicy(max_attempts=1),
            )
            with self.assertRaisesRegex(AvailabilityIndexError, "multiple locations"):
                run_availability_index(config)
            receipt = json.loads(
                (root / "output/availability_receipt.json").read_text()
            )
            self.assertTrue(receipt["complete"])
            self.assertFalse(receipt["integrity_pass"])
            self.assertEqual(
                receipt["duplicates"]["within_route_split_duplicate_video_ids"], 1
            )
            self.assertFalse(
                (root / "output/audioset_availability_index.jsonl").exists()
            )
            self.assertEqual(
                len(
                    _read_jsonl(
                        root / "output/audioset_availability_index.partial.jsonl"
                    )
                ),
                2,
            )

    def test_cli_bounded_run_resumes_to_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_shard(
                root / "mirror/train-000.parquet",
                [
                    ("a", ["/a"], ["A"]),
                    ("b", ["/b"], ["B"]),
                    ("c", ["/c"], ["C"]),
                ],
                row_group_size=2,
            )
            allowlist_path = root / "allowlist.jsonl"
            allowlist_path.write_text(
                "".join(
                    json.dumps({"video_id": value}) + "\n"
                    for value in ("a", "c", "missing")
                ),
                encoding="utf-8",
            )
            common = [
                "--output-dir",
                str(root / "output"),
                "--source-route",
                "local_route",
                "--hf-dataset",
                "local/dataset",
                "--resolved-revision",
                "local-revision",
                "--parquet-pattern",
                str(root / "mirror/{split}-*.parquet"),
                "--split",
                "train",
                "--max-retries",
                "1",
                "--allowlist",
                str(allowlist_path),
            ]
            self.assertEqual(
                availability_main([*common, "--max-new-row-groups", "1"]), 2
            )
            self.assertEqual(availability_main(common), 0)
            receipt = json.loads(
                (root / "output/availability_receipt.json").read_text()
            )
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["mirror_rows_scanned"], 3)
            self.assertEqual(receipt["indexed_allowlist_matches"], 2)
            self.assertEqual(
                receipt["allowlist_audit"]["unmatched_scope_video_ids"], 1
            )


if __name__ == "__main__":
    unittest.main()
