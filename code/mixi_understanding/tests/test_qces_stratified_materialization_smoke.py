from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mixi_understanding.qces.stratified_materialization_smoke import (
    StratifiedSmokeError,
    aggregate_scan_receipts,
    audit_remote_smoke_manifests,
    select_stratified_smoke,
)
from mixi_understanding.scripts.build_qces_stratified_materialization_smoke import (
    main as build_main,
)


def _row(
    *,
    route: str,
    split: str,
    video_id: str,
    label: str,
    tier: int,
) -> dict[str, object]:
    hf_split = "train" if split == "train" else "test"
    dataset = f"dataset/{route}"
    revision = f"revision-{route}"
    return {
        "format": "qces_joint_acoustic_materialization_plan_v1",
        "source_route": route,
        "hf_dataset": dataset,
        "hf_revision": revision,
        "metadata_split": split,
        "hf_split": hf_split,
        "availability_location": {
            "source_route": route,
            "hf_dataset": dataset,
            "hf_revision": revision,
            "hf_split": hf_split,
            "video_id": video_id,
            "parquet_url": f"https://example.invalid/{route}/{hf_split}.parquet",
            "row_group": int(video_id[-1]),
            "row_index": 0,
        },
        "video_id": video_id,
        "split_lock": "test" if split == "eval" else "unassigned_train_pool",
        "events": [
            {
                "label": label,
                "display_name": label.replace("_", " "),
                "mid": f"/m/{label}",
                "onset_seconds": 0.5,
                "offset_seconds": 1.5,
                "segment_id": f"{video_id}_0",
                "video_id": video_id,
            }
        ],
        "crop_requests": [
            {
                "selection_key": f"selection-{video_id}",
                "coverage_label": label,
                "coverage_mid": f"/m/{label}",
                "coverage_rank": 1,
                "ambiguity_tier": tier,
                "ambiguity_tier_name": f"tier_{tier}",
                "crop_start_seconds": 0.25,
                "crop_end_seconds": 1.75,
                "event_onset_seconds": 0.5,
                "event_offset_seconds": 1.5,
            }
        ],
    }


def _scan_receipt(
    *, route_output: dict[str, object], scale: int
) -> dict[str, object]:
    return {
        "scan_only": True,
        "require_preindexed_locations": True,
        "plan_rows": route_output["rows"],
        "plan_files": [{"sha256": route_output["sha256"]}],
        "hf_dataset": route_output["hf_dataset"],
        "resolved_revision": route_output["hf_revision"],
        "preindexed_location_summary": {
            "validated_plan_rows": route_output["rows"]
        },
        "plan_rows_unaccounted_for": 0,
        "disk_preflight": {"safe": True},
        "estimates": {
            "video_id_scan_compressed_column_bytes_estimate": 10 * scale,
            "matching_audio_row_groups_compressed_column_bytes_estimate": 20 * scale,
            "total_compressed_column_bytes_estimate": 30 * scale,
            "total_full_shard_payload_bytes_no_retry_ceiling": 40 * scale,
            "planned_audio_disk_bytes_estimate": 50 * scale,
        },
    }


def _remote_transaction(
    row: dict[str, object], *, sample_rate: int
) -> dict[str, object]:
    request = row["crop_requests"][0]
    route = str(row["source_route"])
    provenance = {
        "transport": "hf_parquet_row_group",
        "source_route": route,
        "hf_dataset": row["hf_dataset"],
        "hf_revision": row["hf_revision"],
    }
    crop = {
        "materialization_item_id": request["selection_key"],
        "sample_rate": sample_rate,
        "coverage_event_retained_fraction": 1.0,
        "coverage_label": request["coverage_label"],
        "coverage_mid": request["coverage_mid"],
        "all_strong_events": [
            {
                "label": request["coverage_label"],
                "audioset_mid": request["coverage_mid"],
                "onset_seconds": 0.25,
                "offset_seconds": 1.25,
            }
        ],
        "source_provenance": provenance,
    }
    return {
        "video_id": row["video_id"],
        "crop_item_count": 1,
        "crop_records": [crop],
        "source_provenance": provenance,
    }


class StratifiedMaterializationSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            _row(
                route="route_a",
                split="train",
                video_id="a0",
                label="Camera",
                tier=0,
            ),
            _row(
                route="route_a",
                split="eval",
                video_id="a1",
                label="Slam",
                tier=1,
            ),
            _row(
                route="route_a",
                split="train",
                video_id="a2",
                label="Printer",
                tier=3,
            ),
            _row(
                route="route_b",
                split="train",
                video_id="b0",
                label="Meow",
                tier=2,
            ),
            _row(
                route="route_b",
                split="train",
                video_id="b1",
                label="Microwave_oven",
                tier=3,
            ),
        ]

    def test_deterministic_route_split_label_tier_and_known_rate_stratification(self) -> None:
        rates = {"a0": 44_100, "a1": 48_000, "b0": 32_000, "b1": 32_000}
        first, report = select_stratified_smoke(
            self.rows, records_per_route=2, seed=7, known_sample_rates=rates
        )
        second, repeated = select_stratified_smoke(
            reversed(self.rows), records_per_route=2, seed=7, known_sample_rates=rates
        )
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )
        self.assertEqual(report, repeated)
        self.assertTrue(report["audit_passes"])
        self.assertEqual(report["active_routes"], 2)
        self.assertEqual(report["selected_source_videos"], 4)
        self.assertEqual(report["selected_metadata_splits"], ["eval", "train"])
        self.assertTrue(
            {44_100, 48_000}.issubset(report["selected_known_native_sample_rates"])
        )
        self.assertEqual(report["selected_unique_stratification_labels"], 4)
        self.assertEqual(report["train_eval_source_overlap"], 0)
        self.assertEqual(
            {route: value["selected_source_videos"] for route, value in report["route_selection"].items()},
            {"route_a": 2, "route_b": 2},
        )

    def test_without_decoded_manifest_never_infers_sample_rate(self) -> None:
        _, report = select_stratified_smoke(self.rows, records_per_route=1, seed=11)
        self.assertTrue(report["audit_passes"])
        self.assertEqual(report["known_target_native_sample_rates"], [])
        self.assertEqual(report["selected_known_native_sample_rates"], [])
        self.assertTrue(report["invariants"]["sample_rate_never_inferred_from_route"])

    def test_conflicting_preindexed_location_fails_closed(self) -> None:
        row = self.rows[0]
        row["availability_location"] = {
            **row["availability_location"],
            "video_id": "wrong",
        }
        with self.assertRaisesRegex(StratifiedSmokeError, "video_id mismatch"):
            select_stratified_smoke([row])

    def test_cli_writes_explicit_combined_and_route_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "joint.jsonl"
            input_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows),
                encoding="utf-8",
            )
            output = root / "output"
            code = build_main(
                [
                    "--plan",
                    str(input_path),
                    "--output-dir",
                    str(output),
                    "--records-per-route",
                    "2",
                ]
            )
            self.assertEqual(code, 0)
            receipt = json.loads(
                (output / "stratified_smoke_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["outputs"]["combined_manifest"]["rows"], 4)
            self.assertEqual(set(receipt["route_outputs"]), {"route_a", "route_b"})
            for value in receipt["route_outputs"].values():
                self.assertTrue(Path(value["path"]).is_file())
                self.assertLessEqual(value["rows"], 2)

    def test_scan_costs_validate_contract_and_sum_routes(self) -> None:
        smoke = {
            "route_outputs": {
                "route_a": {
                    "slug": "route_a",
                    "rows": 2,
                    "sha256": "aaa",
                    "hf_dataset": "dataset/a",
                    "hf_revision": "rev-a",
                },
                "route_b": {
                    "slug": "route_b",
                    "rows": 1,
                    "sha256": "bbb",
                    "hf_dataset": "dataset/b",
                    "hf_revision": "rev-b",
                },
            }
        }
        receipts = {
            route: _scan_receipt(route_output=value, scale=index)
            for index, (route, value) in enumerate(smoke["route_outputs"].items(), start=1)
        }
        report = aggregate_scan_receipts(
            smoke_receipt=smoke, route_receipts=receipts
        )
        self.assertTrue(report["audit_passes"])
        self.assertEqual(report["totals"]["total_compressed_column_bytes_estimate"], 90)
        receipts["route_a"]["scan_only"] = False
        with self.assertRaisesRegex(StratifiedSmokeError, "not scan-only"):
            aggregate_scan_receipts(smoke_receipt=smoke, route_receipts=receipts)

    def test_remote_audit_forbids_existing_manifest_reuse_and_checks_timestamps(self) -> None:
        selected, report = select_stratified_smoke(
            self.rows, records_per_route=1, seed=3
        )
        report["route_outputs"] = {
            route: {
                "slug": route,
                "hf_dataset": values[0]["hf_dataset"],
                "hf_revision": values[0]["hf_revision"],
                "rows": len(values),
                "crop_requests": len(values),
            }
            for route in {str(row["source_route"]) for row in selected}
            for values in [[value for value in selected if value["source_route"] == route]]
        }
        by_route = {
            route: [
                _remote_transaction(row, sample_rate=44_100 if route == "route_a" else 48_000)
                for row in selected
                if row["source_route"] == route
            ]
            for route in report["route_outputs"]
        }
        audit = audit_remote_smoke_manifests(
            smoke_receipt=report, route_manifests=by_route
        )
        self.assertTrue(audit["audit_passes"])
        self.assertEqual(audit["invariants"]["existing_manifest_reuse"], 0)
        self.assertTrue(audit["invariants"]["coverage_event_and_timestamp_retained"])
        by_route["route_a"][0]["source_provenance"]["transport"] = "existing_manifest_reuse"
        with self.assertRaisesRegex(StratifiedSmokeError, "did not read remote"):
            audit_remote_smoke_manifests(
                smoke_receipt=report, route_manifests=by_route
            )

    def test_orchestrator_is_bounded_preindexed_and_has_no_reuse_flag(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "scripts/run_qces_stratified_audiosep_smoke_template.sh"
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("--require-preindexed-locations", text)
        self.assertIn("--min-free-disk-gib 10", text)
        self.assertIn("--max-new-videos", text)
        self.assertIn("--max-items", text)
        self.assertNotIn("--existing-manifest", text)
        self.assertIn("QCES_ALLOW_SMOKE_AUDIO_DOWNLOADS", text)


if __name__ == "__main__":
    unittest.main()
