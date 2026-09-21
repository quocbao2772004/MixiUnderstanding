from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mixi_understanding.data.qces_real10_schema import (
    INFERENCE_FIELDS,
    SCORING_FIELDS,
    canonical_inference_manifest_fingerprint,
    inference_record_fingerprint,
    parse_inference_manifest,
    parse_scoring_manifest,
    project_scoring_manifest,
)
from mixi_understanding.data.qces_v5_tacos import (
    PACKET_FORMAT,
    TacosAuditError,
    canonical_json_sha256,
    sha256_file,
)
from mixi_understanding.data.qces_v5_tacos_audio import AUDIO_RECEIPT_FORMAT
from mixi_understanding.data.qces_v5_tacos_finalize import (
    ADJUDICATION_AUDIT_FORMAT,
    FINALIZATION_FORMAT,
    build_real10_manifests,
    build_scene_questions,
    canonical_audio_manifest_fingerprint,
    canonical_scoring_manifest_fingerprint,
    finalize_human_annotations,
    resolve_rater_pair,
    scene_from_adjudicator,
    write_finalization_artifacts,
)
from mixi_understanding.scripts.finalize_qces_v5_tacos_annotations import (
    OUTPUT_NAMES,
    run as finalize_cli_run,
)


def task() -> dict:
    return {
        "schema_version": PACKET_FORMAT,
        "scene_id": "tacos_123",
        "selection_partition": "real_test",
        "selection_tier": "core",
        "selection_ordinal": 0,
        "source": {
            "freesound_id": "123",
            "filename": "123.mp3",
            "creator_id": "creator-123",
            "clip_duration_seconds": 10.0,
            "upstream_tacos_split": "test",
            "custom_qces_partition": "real_test",
        },
        "audio": {
            "archive_member": "123.mp3",
            "benchmark_window_start_sample_32k": 0,
            "benchmark_window_end_sample_32k": 320_000,
            "benchmark_sample_rate_hz": 32_000,
            "local_path": None,
            "local_sha256": None,
            "verified_audio_properties": None,
        },
        "benchmark_window": {
            "duration_seconds": 10.0,
            "sample_rate_hz": 32_000,
            "start_sample": 0,
            "end_sample": 320_000,
            "selected_before_human_annotation": True,
            "human_labels_used": False,
            "qces_method_outputs_used": False,
            "runtime_or_question_specific": False,
        },
        "proposal_regions": [
            {
                "region_id": f"region_{index:03d}",
                "onset_seconds": float(index * 2),
                "offset_seconds": float(index * 2 + 1),
                "upstream_clip_onset_seconds": float(index * 2),
                "upstream_clip_offset_seconds": float(index * 2 + 1),
                "eligible_proposal": True,
                "selected_chain_member": True,
                "truncated_by_benchmark_window": False,
                "caption": f"immutable TACOS proposal {index}",
            }
            for index in range(4)
        ],
        "suggested_distinct_onset_chain": [f"region_{index:03d}" for index in range(4)],
        "question_contract": {
            "absent_anchor_candidates": [
                "immutable TACOS proposal 0",
                "immutable TACOS proposal 1",
                "immutable TACOS proposal 2",
                "immutable TACOS proposal 3",
                "ancillary TACOS proposal A",
                "ancillary TACOS proposal B",
            ],
            "absent_anchor_cross_scene_support": [
                {
                    "caption": caption,
                    "cross_scene_support_scene_ids": ["tacos_999"],
                    "cross_scene_support_count_↑": 1,
                }
                for caption in [
                    "immutable TACOS proposal 0",
                    "immutable TACOS proposal 1",
                    "immutable TACOS proposal 2",
                    "immutable TACOS proposal 3",
                    "ancillary TACOS proposal A",
                    "ancillary TACOS proposal B",
                ]
            ],
        },
        "audio_receipt_local_path": "upstream/audio/123.wav",
        "audio_receipt_sha256": "a" * 64,
    }


def response(rater: str) -> dict:
    return {
        "rater_id": rater,
        "scene_decision": "accept",
        "selected_region_ids": [f"region_{index:03d}" for index in range(4)],
        "absent_anchor_judgments": [
            {"caption": caption, "confirmed_inaudible": "yes"}
            for caption in [
                "immutable TACOS proposal 0",
                "immutable TACOS proposal 1",
                "immutable TACOS proposal 2",
                "immutable TACOS proposal 3",
                "ancillary TACOS proposal A",
                "ancillary TACOS proposal B",
            ]
        ],
        "regions": [
            {
                "region_id": f"region_{index:03d}",
                "proposal_caption_accurate": "yes",
                "canonical_event_phrase": phrase,
                "verified_onset_seconds": float(index * 2),
                "verified_offset_seconds": float(index * 2 + 1),
                "contamination_rating": "mixed",
                "salience_1_to_5": 4,
            }
            for index, phrase in enumerate(
                ["dog bark", "glass break", "car horn", "child laugh"]
            )
        ],
    }


def partition_task(
    scene_id: str,
    partition: str,
    *,
    source_id: str,
    creator_id: str,
    support_scene_id: str = "tacos_999",
) -> dict:
    value = task()
    value["scene_id"] = scene_id
    value["selection_partition"] = partition
    value["source"] = {
        **value["source"],
        "freesound_id": source_id,
        "creator_id": creator_id,
        "custom_qces_partition": partition,
    }
    for support in value["question_contract"]["absent_anchor_cross_scene_support"]:
        support["cross_scene_support_scene_ids"] = [support_scene_id]
        support["cross_scene_support_count_↑"] = 1
    return value


def writer_fixture(
    root: Path,
) -> tuple[Path, Path, list[dict], list[dict], list[dict], dict]:
    project_root = root / "project"
    output_dir = project_root / "release"
    source_root = project_root / "derived"
    source_root.mkdir(parents=True)

    def make_scene(scene_id: str, split: str, payload: bytes) -> dict:
        source = source_root / f"{scene_id}.wav"
        source.write_bytes(payload)
        bound_task = partition_task(
            scene_id,
            split,
            source_id=f"source-{scene_id}",
            creator_id=f"creator-{scene_id}",
        )
        bound_task["audio_receipt_local_path"] = source.relative_to(
            project_root
        ).as_posix()
        bound_task["audio_receipt_sha256"] = sha256_file(source)
        scene, conflicts = resolve_rater_pair(
            response("rater_a"),
            response("rater_b"),
            task=bound_task,
            maximum_boundary_difference_seconds=0.25,
            minimum_interval_iou=0.80,
        )
        if conflicts or scene is None:
            raise AssertionError(f"invalid writer fixture: {conflicts}")
        return scene

    scenes = [
        make_scene("tacos_123", "real_test", b"canonical-wav-fixture-bytes"),
        make_scene("tacos_124", "real_dev", b"canonical-wav-fixture-bytes-dev"),
    ]
    qa_rows = []
    for ordinal, scene in enumerate(scenes):
        qa_rows.extend(build_scene_questions(scene, global_scene_ordinal=ordinal))
    audit_rows = [
        {"format": ADJUDICATION_AUDIT_FORMAT, "scene_id": scene["scene_id"]}
        for scene in scenes
    ]
    inference_rows, scoring_rows = build_real10_manifests(qa_rows)
    compliance = {
        "format": FINALIZATION_FORMAT,
        "scene_manifest_fingerprint": canonical_json_sha256(scenes),
        "qa_manifest_fingerprint": canonical_json_sha256(qa_rows),
        "adjudication_audit_fingerprint": canonical_json_sha256(audit_rows),
        "qces_real10_inference_manifest_fingerprint": (
            canonical_inference_manifest_fingerprint(
                parse_inference_manifest(inference_rows)
            )
        ),
        "qces_real10_scoring_manifest_fingerprint": (
            canonical_scoring_manifest_fingerprint(scoring_rows)
        ),
        "qces_real10_audio_manifest_fingerprint": (
            canonical_audio_manifest_fingerprint(inference_rows)
        ),
        "qces_real10_real_dev_inference_manifest_fingerprint": (
            canonical_inference_manifest_fingerprint(
                parse_inference_manifest(
                    [row for row in inference_rows if row["split"] == "real_dev"]
                )
            )
        ),
        "qces_real10_real_test_inference_manifest_fingerprint": (
            canonical_inference_manifest_fingerprint(
                parse_inference_manifest(
                    [row for row in inference_rows if row["split"] == "real_test"]
                )
            )
        ),
        "qces_real10_split_inference_union_is_exact_full_manifest": True,
        "qces_real10_inference_is_exact_scoring_projection": True,
    }
    return project_root, output_dir, scenes, qa_rows, audit_rows, compliance


class TacosPairResolutionTest(unittest.TestCase):
    def test_exact_semantics_and_close_boundaries_resolve(self) -> None:
        left = response("rater_a")
        right = response("rater_b")
        right["regions"][0]["verified_onset_seconds"] = 0.05
        scene, conflicts = resolve_rater_pair(
            left,
            right,
            task=task(),
            maximum_boundary_difference_seconds=0.25,
            minimum_interval_iou=0.80,
        )
        self.assertEqual(conflicts, [])
        self.assertIsNotNone(scene)
        assert scene is not None
        self.assertEqual(
            scene["human_verification"]["resolved_by"],
            "independent_pair_agreement",
        )
        self.assertFalse(scene["human_verification"]["model_outputs_visible"])
        self.assertEqual(
            [event["query_event_phrase"] for event in scene["events"]],
            [f"immutable TACOS proposal {index}" for index in range(4)],
        )
        self.assertTrue(
            all(
                event["query_event_phrase_human_verified_accurate"]
                for event in scene["events"]
            )
        )

    def test_absent_anchor_is_selected_only_from_independent_yes_intersection(
        self,
    ) -> None:
        left = response("rater_a")
        right = response("rater_b")
        right["absent_anchor_judgments"][0]["confirmed_inaudible"] = "no"
        scene, conflicts = resolve_rater_pair(
            left,
            right,
            task=task(),
            maximum_boundary_difference_seconds=0.25,
            minimum_interval_iou=0.80,
        )
        self.assertEqual(conflicts, [])
        assert scene is not None
        self.assertEqual(
            scene["selected_absent_anchor_query_phrase"],
            "immutable TACOS proposal 1",
        )
        self.assertNotIn(
            "immutable TACOS proposal 0",
            {
                entry["caption"]
                for entry in scene["jointly_verified_absent_anchor_candidates"]
            },
        )

        for judgment in right["absent_anchor_judgments"]:
            judgment["confirmed_inaudible"] = "no"
        scene, conflicts = resolve_rater_pair(
            left,
            right,
            task=task(),
            maximum_boundary_difference_seconds=0.25,
            minimum_interval_iou=0.80,
        )
        self.assertIsNone(scene)
        self.assertIn("no_jointly_verified_absent_anchor", conflicts)

    def test_semantic_or_large_boundary_disagreement_requires_adjudication(
        self,
    ) -> None:
        left = response("rater_a")
        right = response("rater_b")
        right["regions"][1]["canonical_event_phrase"] = "window shatter"
        scene, conflicts = resolve_rater_pair(
            left,
            right,
            task=task(),
            maximum_boundary_difference_seconds=0.25,
            minimum_interval_iou=0.80,
        )
        self.assertIsNone(scene)
        self.assertIn("phrase_disagrees:1", conflicts)

        right = response("rater_b")
        right["regions"][2]["verified_onset_seconds"] += 0.5
        scene, conflicts = resolve_rater_pair(
            left,
            right,
            task=task(),
            maximum_boundary_difference_seconds=0.25,
            minimum_interval_iou=0.80,
        )
        self.assertIsNone(scene)
        self.assertTrue(any("boundary_disagrees" in value for value in conflicts))

    def test_nonfinite_or_negative_thresholds_fail_closed(self) -> None:
        for maximum_boundary_difference_seconds, minimum_interval_iou in (
            (math.nan, 0.80),
            (math.inf, 0.80),
            (-0.01, 0.80),
            (0.25, math.nan),
            (0.25, 1.01),
        ):
            with self.subTest(
                maximum_boundary_difference_seconds=(
                    maximum_boundary_difference_seconds
                ),
                minimum_interval_iou=minimum_interval_iou,
            ):
                with self.assertRaises(TacosAuditError):
                    resolve_rater_pair(
                        response("rater_a"),
                        response("rater_b"),
                        task=task(),
                        maximum_boundary_difference_seconds=(
                            maximum_boundary_difference_seconds
                        ),
                        minimum_interval_iou=minimum_interval_iou,
                    )

    def test_adjudicator_must_preserve_verified_onset_order(self) -> None:
        adjudication = response("adjudicator")
        adjudication["regions"][0]["verified_onset_seconds"] = 1.0
        adjudication["regions"][0]["verified_offset_seconds"] = 1.8
        adjudication["regions"][1]["verified_onset_seconds"] = 0.5
        adjudication["regions"][1]["verified_offset_seconds"] = 0.9
        with self.assertRaisesRegex(TacosAuditError, "onset order"):
            scene_from_adjudicator(
                adjudication,
                task=task(),
                conflict_reasons=["phrase_disagrees:0"],
            )


class TacosRealQuestionGenerationTest(unittest.TestCase):
    def test_eight_questions_match_synthetic_relation_and_negative_contract(
        self,
    ) -> None:
        scene, conflicts = resolve_rater_pair(
            response("rater_a"),
            response("rater_b"),
            task=task(),
            maximum_boundary_difference_seconds=0.25,
            minimum_interval_iou=0.80,
        )
        self.assertEqual(conflicts, [])
        assert scene is not None
        rows = build_scene_questions(scene, global_scene_ordinal=0)
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row["no_evidence"] for row in rows), 2)
        self.assertEqual(
            {row["relation"] for row in rows}, {"after", "before", "first"}
        )
        for row in rows:
            self.assertEqual(len(row["answer_options"]), 5)
            self.assertEqual(len(set(row["answer_options"])), 5)
            self.assertEqual(
                row["answer_options"][row["answer_option_index"]], row["answer"]
            )
            self.assertFalse(row["clean_reference_stems_available"])
            self.assertFalse(row["waveform_sdr_evaluation_allowed"])
        negatives = [row for row in rows if row["no_evidence"]]
        self.assertTrue(all(row["answer"] == "no_evidence" for row in negatives))
        self.assertTrue(
            all(row["no_evidence_reason"] == "absent_anchor" for row in negatives)
        )
        self.assertTrue(all(not row["evidence_event_ids"] for row in negatives))
        # Positive and absent-anchor controls use the identical surface once
        # the scene-specific anchor phrase is abstracted.  Otherwise the
        # no-evidence head could solve the benchmark from template wording.
        for relation in ("after", "before"):
            relation_rows = [row for row in rows if row["relation"] == relation]
            surfaces = {
                row["question"].replace(row["query_label"], "<anchor>")
                for row in relation_rows
            }
            self.assertEqual(len(surfaces), 1)
        first = [row for row in rows if row["relation"] == "first"]
        self.assertNotEqual(first[0]["question"], first[1]["question"])
        self.assertEqual(first[0]["answer"], first[1]["answer"])
        self.assertIn('"immutable TACOS proposal 0"', first[0]["question"])
        self.assertNotIn('"dog bark"', first[0]["question"])
        positive_after = next(
            row for row in rows if row["relation"] == "after" and not row["no_evidence"]
        )
        self.assertEqual(positive_after["query_label"], "immutable TACOS proposal 0")
        self.assertIn('"immutable TACOS proposal 0"', positive_after["question"])
        self.assertNotIn('"dog bark"', positive_after["question"])

    def test_authoritative_views_are_exact_bound_projections(self) -> None:
        scene, conflicts = resolve_rater_pair(
            response("rater_a"),
            response("rater_b"),
            task=task(),
            maximum_boundary_difference_seconds=0.25,
            minimum_interval_iou=0.80,
        )
        self.assertEqual(conflicts, [])
        assert scene is not None
        qa_rows = build_scene_questions(scene, global_scene_ordinal=0)
        inference_rows, scoring_rows = build_real10_manifests(qa_rows)
        inference = parse_inference_manifest(inference_rows)
        scoring = parse_scoring_manifest(scoring_rows)

        self.assertEqual(len(inference), 8)
        self.assertEqual(len(scoring), 8)
        self.assertTrue(
            all(set(row) == set(INFERENCE_FIELDS) for row in inference_rows)
        )
        self.assertTrue(all(set(row) == set(SCORING_FIELDS) for row in scoring_rows))
        self.assertEqual(
            [record.to_dict() for record in inference],
            [record.to_dict() for record in project_scoring_manifest(scoring)],
        )
        self.assertEqual(
            {row["mixture_path"] for row in inference_rows}, {"audio/tacos_123.wav"}
        )
        self.assertEqual({row["mixture_sha256"] for row in inference_rows}, {"a" * 64})
        self.assertEqual(
            {row["question_index"] for row in inference_rows}, set(range(8))
        )
        self.assertTrue(
            all(
                row["inference_record_sha256"]
                == inference_record_fingerprint(inference[index])
                for index, row in enumerate(scoring_rows)
            )
        )
        forbidden = {
            "events",
            "source",
            "evidence_stem_path",
            "oracle_evidence_path",
            "si_sdr",
            "sd_sdr",
        }
        self.assertTrue(all(forbidden.isdisjoint(row) for row in scoring_rows))
        self.assertTrue(all("anchor_intervals" in row for row in scoring_rows))
        self.assertTrue(all("answer_intervals" in row for row in scoring_rows))


class TacosFinalizationAuditTest(unittest.TestCase):
    def test_overlap_is_computed_and_written_to_disagreement_audit(self) -> None:
        tasks = [
            partition_task(
                "tacos_101",
                "real_dev",
                source_id="shared-source",
                creator_id="shared-creator",
                support_scene_id="tacos_102",
            ),
            partition_task(
                "tacos_102",
                "real_dev",
                source_id="dev-source-102",
                creator_id="dev-creator-102",
                support_scene_id="tacos_101",
            ),
            partition_task(
                "tacos_201",
                "real_test",
                source_id="shared-source",
                creator_id="shared-creator",
                support_scene_id="tacos_202",
            ),
            partition_task(
                "tacos_202",
                "real_test",
                source_id="test-source-202",
                creator_id="test-creator-202",
                support_scene_id="tacos_201",
            ),
        ]

        def read_response(**kwargs: object) -> dict:
            return response(str(kwargs["rater_id"]))

        with patch(
            "mixi_understanding.data.qces_v5_tacos_finalize.load_bound_tasks",
            return_value=("packet-fingerprint", "audio-fingerprint", tasks),
        ), patch(
            "mixi_understanding.data.qces_v5_tacos_finalize._read_response",
            side_effect=read_response,
        ):
            compliance, scenes, qa_rows, audit_rows = finalize_human_annotations(
                packet_path=Path("packet.jsonl"),
                audio_receipt_path=Path("receipt.jsonl"),
                response_root=Path("responses"),
                rater_a_id="rater_a",
                rater_b_id="rater_b",
                adjudicator_id=None,
                project_root=Path("."),
                target_real_dev_scenes=2,
                target_real_test_scenes=2,
                maximum_boundary_difference_seconds=0.25,
                minimum_interval_iou=0.80,
            )

        metrics = compliance["metrics"]
        self.assertEqual(metrics["final_dev_test_source_overlap_↓"], 1)
        self.assertEqual(metrics["final_dev_test_creator_overlap_↓"], 1)
        self.assertFalse(compliance["submission_real_data_gate_passed"])
        self.assertIn("real_dev_test_source_overlap", compliance["gate_failures"])
        self.assertIn("real_dev_test_creator_overlap", compliance["gate_failures"])
        self.assertEqual(len(scenes), 4)
        self.assertEqual(len(qa_rows), 32)
        self.assertEqual(len(audit_rows), 4)
        self.assertTrue(all(row["included_in_final_manifest"] for row in audit_rows))
        self.assertTrue(
            all(
                row["resolution_status"] == "resolved_by_independent_pair"
                for row in audit_rows
            )
        )
        self.assertEqual(
            compliance["adjudication_audit_fingerprint"],
            canonical_json_sha256(audit_rows),
        )
        inference_rows, scoring_rows = build_real10_manifests(qa_rows)
        self.assertEqual(compliance["format"], FINALIZATION_FORMAT)
        self.assertEqual(
            compliance["input_contract"],
            {
                "annotation_packet_schema_version": PACKET_FORMAT,
                "derived_audio_receipt_format": AUDIO_RECEIPT_FORMAT,
            },
        )
        self.assertTrue(compliance["qces_real10_inference_is_exact_scoring_projection"])
        self.assertEqual(
            compliance["qces_real10_scoring_manifest_fingerprint"],
            canonical_scoring_manifest_fingerprint(scoring_rows),
        )
        self.assertEqual(
            compliance["qces_real10_audio_manifest_fingerprint"],
            canonical_audio_manifest_fingerprint(inference_rows),
        )

    def test_conflict_and_third_rater_resolution_are_auditable(self) -> None:
        tasks = [
            partition_task(
                "tacos_101",
                "real_dev",
                source_id="source-101",
                creator_id="creator-101",
                support_scene_id="tacos_102",
            ),
            partition_task(
                "tacos_102",
                "real_dev",
                source_id="source-102",
                creator_id="creator-102",
                support_scene_id="tacos_101",
            ),
            partition_task(
                "tacos_201",
                "real_test",
                source_id="source-201",
                creator_id="creator-201",
                support_scene_id="tacos_202",
            ),
            partition_task(
                "tacos_202",
                "real_test",
                source_id="source-202",
                creator_id="creator-202",
                support_scene_id="tacos_201",
            ),
        ]

        def read_response(**kwargs: object) -> dict:
            result = response(str(kwargs["rater_id"]))
            if kwargs["rater_id"] == "rater_b" and kwargs["task"] == tasks[2]:
                result["regions"][0]["canonical_event_phrase"] = "different event"
            return result

        with patch(
            "mixi_understanding.data.qces_v5_tacos_finalize.load_bound_tasks",
            return_value=("packet-fingerprint", "audio-fingerprint", tasks),
        ), patch(
            "mixi_understanding.data.qces_v5_tacos_finalize._read_response",
            side_effect=read_response,
        ):
            compliance, _, _, audit_rows = finalize_human_annotations(
                packet_path=Path("packet.jsonl"),
                audio_receipt_path=Path("receipt.jsonl"),
                response_root=Path("responses"),
                rater_a_id="rater_a",
                rater_b_id="rater_b",
                adjudicator_id="adjudicator",
                project_root=Path("."),
                target_real_dev_scenes=2,
                target_real_test_scenes=2,
                maximum_boundary_difference_seconds=0.25,
                minimum_interval_iou=0.80,
            )

        adjudicated = audit_rows[2]
        self.assertEqual(adjudicated["pair_conflicts"], ["phrase_disagrees:0"])
        self.assertTrue(adjudicated["adjudicator_response_present"])
        self.assertEqual(adjudicated["adjudicator_scene_decision"], "accept")
        self.assertEqual(
            adjudicated["resolution_status"],
            "resolved_by_third_rater_adjudication",
        )
        self.assertEqual(compliance["metrics"]["resolved_pair_scenes_↑"], 3)
        self.assertEqual(compliance["metrics"]["resolved_total_scenes_↑"], 4)
        self.assertEqual(compliance["metrics"]["third_rater_adjudications_↓"], 1)

    def test_empty_targets_and_invalid_rater_ids_fail_closed(self) -> None:
        common = {
            "packet_path": Path("packet.jsonl"),
            "audio_receipt_path": Path("receipt.jsonl"),
            "response_root": Path("responses"),
            "rater_a_id": "rater_a",
            "rater_b_id": "rater_b",
            "adjudicator_id": None,
            "project_root": Path("."),
            "target_real_dev_scenes": 1,
            "target_real_test_scenes": 1,
            "maximum_boundary_difference_seconds": 0.25,
            "minimum_interval_iou": 0.80,
        }
        with self.assertRaisesRegex(TacosAuditError, "positive integers"):
            finalize_human_annotations(**{**common, "target_real_dev_scenes": 0})
        with self.assertRaisesRegex(TacosAuditError, "rater IDs"):
            finalize_human_annotations(**{**common, "rater_a_id": "invalid id"})

    def test_finalizer_rejects_noncanonical_current_packet_task(self) -> None:
        invalid = task()
        invalid["benchmark_window"]["end_sample"] = 319_999
        with patch(
            "mixi_understanding.data.qces_v5_tacos_finalize.load_bound_tasks",
            return_value=("packet-fingerprint", "audio-fingerprint", [invalid]),
        ):
            with self.assertRaisesRegex(TacosAuditError, "packet 10 s"):
                finalize_human_annotations(
                    packet_path=Path("packet.jsonl"),
                    audio_receipt_path=Path("receipt.jsonl"),
                    response_root=Path("responses"),
                    rater_a_id="rater_a",
                    rater_b_id="rater_b",
                    adjudicator_id=None,
                    project_root=Path("."),
                    target_real_dev_scenes=1,
                    target_real_test_scenes=1,
                    maximum_boundary_difference_seconds=0.25,
                    minimum_interval_iou=0.80,
                )

    def test_finalizer_loader_fails_closed_on_old_packet_or_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet_path = root / "packet.jsonl"
            receipt_path = root / "receipt.jsonl"
            packet_path.write_text(
                json.dumps({"schema_version": "qces_v5_tacos_annotation_packet_v1"})
                + "\n",
                encoding="utf-8",
            )
            receipt_path.write_text(
                json.dumps({"format": AUDIO_RECEIPT_FORMAT}) + "\n",
                encoding="utf-8",
            )
            common = {
                "packet_path": packet_path,
                "audio_receipt_path": receipt_path,
                "response_root": root / "responses",
                "rater_a_id": "rater_a",
                "rater_b_id": "rater_b",
                "adjudicator_id": None,
                "project_root": root,
                "target_real_dev_scenes": 1,
                "target_real_test_scenes": 1,
                "maximum_boundary_difference_seconds": 0.25,
                "minimum_interval_iou": 0.80,
            }
            with self.assertRaisesRegex(TacosAuditError, "packet schema mismatch"):
                finalize_human_annotations(**common)

            packet_path.write_text(
                json.dumps(task()) + "\n",
                encoding="utf-8",
            )
            receipt_path.write_text(
                json.dumps(
                    {
                        "format": "qces_v5_tacos_derived_audio_receipt_v1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TacosAuditError, "receipt schema mismatch"):
                finalize_human_annotations(**common)

    def test_artifact_writer_binds_all_rows_and_preflights_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (
                project_root,
                output_dir,
                scenes,
                qa_rows,
                audit_rows,
                compliance,
            ) = writer_fixture(Path(temporary))
            write_finalization_artifacts(
                output_dir=output_dir,
                project_root=project_root,
                compliance=compliance,
                scenes=scenes,
                qa_rows=qa_rows,
                audit_rows=audit_rows,
                overwrite=False,
            )
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "qces_v5_tacos_human_compliance.json",
                    "qces_v5_tacos_verified_scenes.jsonl",
                    "qces_v5_tacos_real_qa.jsonl",
                    "qces_v5_tacos_adjudication_audit.jsonl",
                    "qces_real10_inference.jsonl",
                    "qces_real10_scoring.jsonl",
                    "qces_real10_inference_real_dev.jsonl",
                    "qces_real10_inference_real_test.jsonl",
                    "audio",
                },
            )
            released_audio = output_dir / "audio" / "tacos_123.wav"
            self.assertEqual(
                sha256_file(released_audio),
                qa_rows[0]["audio_sha256"],
            )
            inference_rows = [
                json.loads(line)
                for line in (output_dir / "qces_real10_inference.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            scoring_rows = [
                json.loads(line)
                for line in (output_dir / "qces_real10_scoring.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            dev_inference_rows = [
                json.loads(line)
                for line in (output_dir / "qces_real10_inference_real_dev.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            test_inference_rows = [
                json.loads(line)
                for line in (output_dir / "qces_real10_inference_real_test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            inference = parse_inference_manifest(inference_rows)
            scoring = parse_scoring_manifest(scoring_rows)
            self.assertEqual(
                [record.to_dict() for record in inference],
                [record.to_dict() for record in project_scoring_manifest(scoring)],
            )
            self.assertEqual({row["split"] for row in dev_inference_rows}, {"real_dev"})
            self.assertEqual(
                {row["split"] for row in test_inference_rows}, {"real_test"}
            )
            self.assertEqual(
                sorted(
                    dev_inference_rows + test_inference_rows, key=lambda row: row["id"]
                ),
                sorted(inference_rows, key=lambda row: row["id"]),
            )

            # Exact released audio is reusable even though JSON artifacts are
            # intentionally rewritten under --overwrite.
            write_finalization_artifacts(
                output_dir=output_dir,
                project_root=project_root,
                compliance=compliance,
                scenes=scenes,
                qa_rows=qa_rows,
                audit_rows=audit_rows,
                overwrite=True,
            )

        with tempfile.TemporaryDirectory() as temporary:
            (
                project_root,
                output_dir,
                scenes,
                qa_rows,
                audit_rows,
                compliance,
            ) = writer_fixture(Path(temporary))
            output_dir.mkdir(parents=True)
            (output_dir / "qces_v5_tacos_adjudication_audit.jsonl").write_text(
                "existing\n", encoding="utf-8"
            )
            with self.assertRaises(FileExistsError):
                write_finalization_artifacts(
                    output_dir=output_dir,
                    project_root=project_root,
                    compliance=compliance,
                    scenes=scenes,
                    qa_rows=qa_rows,
                    audit_rows=audit_rows,
                    overwrite=False,
                )
            self.assertFalse(
                (output_dir / "qces_v5_tacos_human_compliance.json").exists()
            )
            self.assertFalse((output_dir / "audio").exists())

        with tempfile.TemporaryDirectory() as temporary:
            (
                project_root,
                output_dir,
                scenes,
                qa_rows,
                audit_rows,
                compliance,
            ) = writer_fixture(Path(temporary))
            mutated = [dict(row) for row in qa_rows]
            mutated[0]["question"] = "A changed but structurally valid question?"
            with self.assertRaisesRegex(TacosAuditError, "qa_manifest_fingerprint"):
                write_finalization_artifacts(
                    output_dir=output_dir,
                    project_root=project_root,
                    compliance=compliance,
                    scenes=scenes,
                    qa_rows=mutated,
                    audit_rows=audit_rows,
                    overwrite=False,
                )
            self.assertFalse(output_dir.exists())

    def test_artifact_writer_never_overwrites_mismatched_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (
                project_root,
                output_dir,
                scenes,
                qa_rows,
                audit_rows,
                compliance,
            ) = writer_fixture(Path(temporary))
            write_finalization_artifacts(
                output_dir=output_dir,
                project_root=project_root,
                compliance=compliance,
                scenes=scenes,
                qa_rows=qa_rows,
                audit_rows=audit_rows,
                overwrite=False,
            )
            released_audio = output_dir / "audio" / "tacos_123.wav"
            released_audio.write_bytes(b"mismatched")
            with self.assertRaisesRegex(TacosAuditError, "refusing to overwrite"):
                write_finalization_artifacts(
                    output_dir=output_dir,
                    project_root=project_root,
                    compliance=compliance,
                    scenes=scenes,
                    qa_rows=qa_rows,
                    audit_rows=audit_rows,
                    overwrite=True,
                )
            self.assertEqual(released_audio.read_bytes(), b"mismatched")

    def test_artifact_writer_accepts_exact_source_equal_to_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (
                project_root,
                output_dir,
                scenes,
                qa_rows,
                audit_rows,
                compliance,
            ) = writer_fixture(Path(temporary))
            original_source = project_root / qa_rows[0]["audio_path"]
            destination = output_dir / "audio" / "tacos_123.wav"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(original_source.read_bytes())
            relative = destination.relative_to(project_root).as_posix()
            scenes[0]["audio"]["local_path"] = relative
            for row in qa_rows:
                if row["scene_id"] == scenes[0]["scene_id"]:
                    row["audio_path"] = relative
            compliance["scene_manifest_fingerprint"] = canonical_json_sha256(scenes)
            compliance["qa_manifest_fingerprint"] = canonical_json_sha256(qa_rows)

            write_finalization_artifacts(
                output_dir=output_dir,
                project_root=project_root,
                compliance=compliance,
                scenes=scenes,
                qa_rows=qa_rows,
                audit_rows=audit_rows,
                overwrite=False,
            )
            self.assertEqual(sha256_file(destination), qa_rows[0]["audio_sha256"])


class TacosFinalizeCliTest(unittest.TestCase):
    def test_cli_preflights_new_manifests_and_passes_project_root(self) -> None:
        self.assertIn("qces_real10_inference.jsonl", OUTPUT_NAMES)
        self.assertIn("qces_real10_scoring.jsonl", OUTPUT_NAMES)
        self.assertIn("qces_real10_inference_real_dev.jsonl", OUTPUT_NAMES)
        self.assertIn("qces_real10_inference_real_test.jsonl", OUTPUT_NAMES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "release"
            args = SimpleNamespace(
                output_dir=output_dir,
                project_root=root,
                annotation_packet=root / "packet.jsonl",
                audio_receipt=root / "receipt.jsonl",
                response_root=root / "responses",
                rater_a_id="rater_a",
                rater_b_id="rater_b",
                adjudicator_id=None,
                target_real_dev_scenes=1,
                target_real_test_scenes=1,
                maximum_boundary_difference_seconds=0.25,
                minimum_interval_iou=0.80,
                overwrite=False,
            )
            compliance = {"submission_real_data_gate_passed": True}
            with patch(
                "mixi_understanding.scripts.finalize_qces_v5_tacos_annotations."
                "finalize_human_annotations",
                return_value=(compliance, [{"scene_id": "x"}], [{"id": "q"}], []),
            ), patch(
                "mixi_understanding.scripts.finalize_qces_v5_tacos_annotations."
                "write_finalization_artifacts"
            ) as writer:
                self.assertEqual(finalize_cli_run(args), 0)
            self.assertEqual(writer.call_args.kwargs["project_root"], root.resolve())

            output_dir.mkdir()
            (output_dir / "qces_real10_scoring.jsonl").write_text(
                "existing\n", encoding="utf-8"
            )
            with self.assertRaises(FileExistsError):
                finalize_cli_run(args)


if __name__ == "__main__":
    unittest.main()
