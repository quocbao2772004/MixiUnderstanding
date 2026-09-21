from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixi_understanding.data.qces_real10_schema import (
    DURATION_SECONDS,
    INFERENCE_SCHEMA_VERSION,
    NUM_CHANNELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    SCORING_SCHEMA_VERSION,
    canonical_inference_manifest_fingerprint,
    canonical_json_sha256,
    inference_record_fingerprint,
    parse_scoring_manifest,
    project_scoring_fields_to_inference,
    project_scoring_manifest,
)
from mixi_understanding.qces.config import (
    DUAL_ROLE_SEMANTIC_MODE,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
)
from mixi_understanding.qces.real10_prediction import (
    METHOD_FREEZE_RECEIPT_FORMAT,
    PREDICTION_SCHEMA_VERSION,
    RENDER_RECEIPT_FORMAT,
    pcm_f32le_sha256,
    prediction_manifest_fingerprint,
    validate_prediction_record,
    write_deterministic_float_wav,
)
from mixi_understanding.qces.real10_scoring import (
    METHOD_IDENTITY_FORMAT,
    REPORT_FORMAT,
    RUN_SPEC_FORMAT,
    STEM_AGGREGATE_FORMAT,
    Real10ScoringError,
    creator_cluster_bootstrap,
    score_qces_real10,
    temporal_iou,
    validate_scoring_inputs,
    write_score_report_exclusive,
)
from mixi_understanding.scripts.score_qces_real10 import main as scoring_cli_main


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scoring_row(
    *,
    scene_index: int,
    question_index: int,
    no_evidence: bool,
    mixture_sha256: str,
    split: str,
) -> dict:
    scene_id = f"scene_{scene_index}"
    relation = "first" if no_evidence else "after"
    row = {
        "schema_version": SCORING_SCHEMA_VERSION,
        "id": f"{scene_id}__q{question_index}",
        "scene_id": scene_id,
        "scene_family_id": scene_id,
        "split": split,
        "question_index": question_index,
        "question_type": f"temporal_{relation}",
        "relation": relation,
        "question": (
            "Which sound starts first?"
            if no_evidence
            else "What sound begins after the anchor?"
        ),
        "sample_rate": SAMPLE_RATE,
        "num_channels": NUM_CHANNELS,
        "num_samples": NUM_SAMPLES,
        "duration_seconds": DURATION_SECONDS,
        "mixture_path": f"audio/{scene_id}.wav",
        "mixture_sha256": mixture_sha256,
        "inference_record_sha256": "0" * 64,
        "answer_options": ["answer", "anchor", "noise", "music", "no_evidence"],
        "answer": "no_evidence" if no_evidence else "answer",
        "answer_option_index": 4 if no_evidence else 0,
        "no_evidence": no_evidence,
        "creator_id": f"creator-{scene_index}",
        "anchor_event_ids": [] if no_evidence else ["anchor_event"],
        "answer_event_ids": [] if no_evidence else ["answer_event"],
        "evidence_event_ids": ([] if no_evidence else ["anchor_event", "answer_event"]),
        "anchor_intervals": [] if no_evidence else [[1.0, 2.0]],
        "answer_intervals": [] if no_evidence else [[3.0, 4.0]],
        "evidence_intervals": ([] if no_evidence else [[1.0, 2.0], [3.0, 4.0]]),
        "upstream_tacos_split": "development" if scene_index == 0 else "test",
        "clean_reference_stems_available": False,
        "waveform_sdr_evaluation_allowed": False,
    }
    row["inference_record_sha256"] = inference_record_fingerprint(
        project_scoring_fields_to_inference(row)
    )
    return row


class Real10ScoringFixture:
    def __init__(
        self,
        root: Path,
        *,
        split: str = "real_dev",
        broken_reconstruction: bool = False,
    ) -> None:
        self.root = root
        self.dataset_root = root / "dataset"
        self.prediction_root = root / "predictions"
        self.dataset_root.mkdir()
        self.prediction_root.mkdir()
        (self.dataset_root / "audio").mkdir()
        self.scoring_manifest = self.dataset_root / "qces_real10_scoring.jsonl"
        self.split = split

        mixtures: dict[str, np.ndarray] = {}
        mixture_specs: dict[str, dict] = {}
        scoring_rows: list[dict] = []
        time = np.arange(NUM_SAMPLES, dtype=np.float64) / SAMPLE_RATE
        for scene_index in range(2):
            samples = (
                0.1 * np.sin(2.0 * np.pi * (220.0 + 30 * scene_index) * time)
            ).astype(np.float32)
            scene_id = f"scene_{scene_index}"
            mixture_path = self.dataset_root / "audio" / f"{scene_id}.wav"
            identity = write_deterministic_float_wav(mixture_path, samples)
            mixture_sha256 = _sha256_file(mixture_path)
            mixtures[scene_id] = samples
            mixture_specs[scene_id] = {
                "scene_id": scene_id,
                "manifest_path": f"audio/{scene_id}.wav",
                "file_sha256": mixture_sha256,
                "pcm_f32le_sha256": pcm_f32le_sha256(samples),
                "size_bytes": identity["size_bytes"],
            }
            scoring_rows.extend(
                [
                    _scoring_row(
                        scene_index=scene_index,
                        question_index=0,
                        no_evidence=False,
                        mixture_sha256=mixture_sha256,
                        split=split,
                    ),
                    _scoring_row(
                        scene_index=scene_index,
                        question_index=1,
                        no_evidence=True,
                        mixture_sha256=mixture_sha256,
                        split=split,
                    ),
                ]
            )
        _jsonl(self.scoring_manifest, scoring_rows)
        scoring_records = parse_scoring_manifest(scoring_rows)

        prediction_rows: list[dict] = []
        for scoring in scoring_records:
            mixture = mixtures[scoring.scene_id]
            if scoring.no_evidence:
                anchor: list[list[float]] = []
                answer: list[list[float]] = []
                union: list[list[float]] = []
                evidence = np.zeros_like(mixture)
                probability = 0.9
            else:
                anchor = [[1.0, 2.0]]
                answer = [[3.0, 4.0]]
                union = [[1.0, 2.0], [3.0, 4.0]]
                mask = np.zeros(NUM_SAMPLES, dtype=np.float32)
                mask[SAMPLE_RATE : 2 * SAMPLE_RATE] = 1.0
                mask[3 * SAMPLE_RATE : 4 * SAMPLE_RATE] = 1.0
                evidence = mixture * mask
                probability = 0.1
            residual = mixture - evidence
            if broken_reconstruction and scoring.sample_id.endswith("__q0"):
                residual = residual.copy()
                residual[0] += np.float32(1e-4)
            reconstruction_error = float(np.max(np.abs(evidence + residual - mixture)))
            evidence_path = (
                self.prediction_root / "stems" / scoring.sample_id / "evidence.wav"
            )
            residual_path = evidence_path.with_name("residual.wav")
            evidence_identity = write_deterministic_float_wav(evidence_path, evidence)
            residual_identity = write_deterministic_float_wav(residual_path, residual)
            evidence_identity["path"] = evidence_path.relative_to(
                self.prediction_root
            ).as_posix()
            residual_identity["path"] = residual_path.relative_to(
                self.prediction_root
            ).as_posix()
            mixture_spec = mixture_specs[scoring.scene_id]
            row = {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "id": scoring.sample_id,
                "scene_id": scoring.scene_id,
                "split": split,
                "question_index": scoring.question_index,
                "relation": scoring.relation,
                "inference_record_sha256": inference_record_fingerprint(
                    scoring.to_inference()
                ),
                "mixture": {
                    "manifest_path": scoring.mixture_path,
                    "file_sha256": mixture_spec["file_sha256"],
                    "pcm_f32le_sha256": mixture_spec["pcm_f32le_sha256"],
                    "size_bytes": mixture_spec["size_bytes"],
                    "sample_rate": SAMPLE_RATE,
                    "num_channels": NUM_CHANNELS,
                    "num_samples": NUM_SAMPLES,
                    "duration_seconds": DURATION_SECONDS,
                },
                "prediction": {
                    "anchor_intervals": anchor,
                    "answer_intervals": answer,
                    "union_intervals": union,
                    "no_evidence_probability": probability,
                    "no_evidence_prediction": probability >= 0.5,
                    "same_semantic_probability": 0.2,
                    "role_threshold": 0.5,
                    "no_evidence_threshold": 0.5,
                    "temporal_role_mode": OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
                    "semantic_separation_mode": DUAL_ROLE_SEMANTIC_MODE,
                },
                "stems": {
                    "evidence": evidence_identity,
                    "residual": residual_identity,
                },
                "separator_calls": {
                    "physical_forwards": 1,
                    "effective_evaluations": 2,
                },
                "max_evidence_plus_residual_minus_mixture_abs_error": (
                    reconstruction_error
                ),
            }
            validate_prediction_record(row)
            prediction_rows.append(row)
        prediction_manifest = self.prediction_root / "prediction_manifest.jsonl"
        _jsonl(prediction_manifest, prediction_rows)

        method = {
            "format": METHOD_IDENTITY_FORMAT,
            "inference_schema_version": INFERENCE_SCHEMA_VERSION,
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "fixture_method": "frozen_audiosep",
        }
        freeze_identity = None
        if split == "real_test":
            freeze_path = root / "method_freeze_receipt.json"
            freeze_payload = {
                "format": METHOD_FREEZE_RECEIPT_FORMAT,
                "purpose": "freeze_method_before_single_real_test_render",
                "source_split": "real_dev",
                "method": method,
                "method_identity_sha256": canonical_json_sha256(method),
                "declaration": {
                    "method_selected_without_real_test_metrics": True,
                    "real_test_model_selection_prohibited": True,
                    "single_real_test_render_after_freeze": True,
                    "real_test_render_count_before_freeze_↓": 0,
                },
            }
            freeze_path.write_text(
                json.dumps(freeze_payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            freeze_identity = {
                "path": str(freeze_path.resolve()),
                "sha256": _sha256_file(freeze_path),
                "size_bytes": freeze_path.stat().st_size,
            }
        projected = project_scoring_manifest(scoring_records)
        selected_ids = sorted(record.sample_id for record in scoring_records)
        run_spec = {
            "format": RUN_SPEC_FORMAT,
            "split": split,
            "source_inference_manifest": {
                "path": str(
                    (self.dataset_root / "qces_real10_inference.jsonl").resolve()
                ),
                "sha256": "a" * 64,
                "size_bytes": 123,
                "canonical_fingerprint": canonical_inference_manifest_fingerprint(
                    projected
                ),
                "record_count": len(scoring_records),
            },
            "selected_inference_view": {
                "canonical_fingerprint": canonical_inference_manifest_fingerprint(
                    projected
                ),
                "record_count": len(scoring_records),
                "scene_count": 2,
                "sample_ids": selected_ids,
            },
            "selected_mixtures": [
                mixture_specs[scene_id] for scene_id in sorted(mixture_specs)
            ],
            "foundation_cache": {"fixture": True},
            "method": method,
            "method_identity_sha256": canonical_json_sha256(method),
            "method_freeze_receipt": freeze_identity,
        }
        maximum_error = max(
            row["max_evidence_plus_residual_minus_mixture_abs_error"]
            for row in prediction_rows
        )
        manifest_identity = {
            "sha256": _sha256_file(prediction_manifest),
            "size_bytes": prediction_manifest.stat().st_size,
        }
        stem_payload = [
            {
                "id": row["id"],
                "evidence": row["stems"]["evidence"],
                "residual": row["stems"]["residual"],
            }
            for row in sorted(prediction_rows, key=lambda item: item["id"])
        ]
        receipt = {
            "format": RENDER_RECEIPT_FORMAT,
            "purpose": "label_free_qces_real10_evidence_and_residual_render",
            "run_spec": run_spec,
            "run_identity_sha256": canonical_json_sha256(run_spec),
            "execution": {
                "device": "cpu",
                "batch_size": 1,
                "precision": "float32_no_autocast",
                "determinism": {
                    "seed": 2026,
                    "torch_deterministic_algorithms": True,
                    "cublas_workspace_config": None,
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                },
                "software": {"fixture": "1"},
            },
            "counts": {
                "prediction_records_↑": len(prediction_rows),
                "unique_scenes_↑": 2,
                "physical_separator_forwards_↓": len(prediction_rows),
                "effective_separator_evaluations_↓": 2 * len(prediction_rows),
                "maximum_evidence_plus_residual_minus_mixture_abs_error_↓": (
                    maximum_error
                ),
            },
            "gates": {
                "maximum_evidence_plus_residual_minus_mixture_abs_error_threshold_↓": (
                    1e-6
                ),
                "observed_maximum_↓": maximum_error,
                "reconstruction_gate_passed": True,
            },
            "artifacts": {
                "prediction_manifest": {
                    "filename": "prediction_manifest.jsonl",
                    **manifest_identity,
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "canonical_fingerprint": prediction_manifest_fingerprint(
                        prediction_rows
                    ),
                },
                "stem_aggregate_sha256": canonical_json_sha256(
                    {"format": STEM_AGGREGATE_FORMAT, "artifacts": stem_payload}
                ),
            },
            "input_boundary": {
                "accepted_manifest_schema": INFERENCE_SCHEMA_VERSION,
                "scoring_or_gold_manifest_opened": False,
                "human_answer_fields_consumed": False,
                "human_temporal_fields_consumed": False,
                "clean_reference_metrics_emitted": False,
            },
        }
        (self.prediction_root / "render_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


class Real10MetricDefinitionTest(unittest.TestCase):
    def test_continuous_interval_union_iou(self) -> None:
        self.assertAlmostEqual(
            temporal_iou([[1.0, 2.0], [2.0, 3.0]], [[2.0, 4.0]]), 1.0 / 3.0
        )
        self.assertEqual(temporal_iou([], [[1.0, 2.0]]), 0.0)
        self.assertEqual(temporal_iou([], []), 1.0)


class Real10EndToEndScoringTest(unittest.TestCase):
    def test_strict_fixture_scores_expected_metrics_and_arrows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10ScoringFixture(Path(temporary))
            report = score_qces_real10(
                scoring_manifest_path=fixture.scoring_manifest,
                dataset_root=fixture.dataset_root,
                prediction_root=fixture.prediction_root,
                bootstrap_replicates=20,
                seed=17,
            )
        self.assertEqual(report["format"], REPORT_FORMAT)
        self.assertTrue(report["integrity_gates"]["all_integrity_gates_passed"])
        metrics = report["overall"]["metrics"]
        self.assertEqual(metrics["anchor_temporal_iou_answerable_↑"], 1.0)
        self.assertEqual(metrics["answer_temporal_iou_answerable_↑"], 1.0)
        self.assertEqual(metrics["union_temporal_iou_answerable_↑"], 1.0)
        self.assertEqual(metrics["weakest_role_temporal_iou_answerable_↑"], 1.0)
        self.assertEqual(metrics["onset_boundary_mae_seconds_answerable_↓"], 0.0)
        self.assertEqual(metrics["no_evidence_auroc_↑"], 1.0)
        self.assertEqual(metrics["no_evidence_auprc_↑"], 1.0)
        self.assertAlmostEqual(metrics["no_evidence_brier_↓"], 0.01)
        self.assertAlmostEqual(metrics["no_evidence_ece_10bin_↓"], 0.1)
        self.assertEqual(metrics["no_evidence_retained_energy_ratio_↓"], 0.0)
        self.assertEqual(metrics["answerable_retained_duration_ratio_↓"], 0.2)
        self.assertEqual(metrics["physical_separator_forwards_per_record_↓"], 1.0)
        self.assertEqual(metrics["effective_separator_evaluations_per_record_↓"], 2.0)
        self.assertEqual(
            report["overall"]["creator_clustered_bootstrap"]["row_bootstrap_used"],
            False,
        )
        self.assertEqual(set(report["slices"]["by_relation"]), {"after", "first"})
        for metric, arrow in report["metric_directions"].items():
            self.assertTrue(metric.endswith(arrow))

    def test_cluster_bootstrap_is_deterministic_and_keeps_scene_rows_together(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10ScoringFixture(Path(temporary))
            validated = validate_scoring_inputs(
                scoring_manifest_path=fixture.scoring_manifest,
                dataset_root=fixture.dataset_root,
                prediction_root=fixture.prediction_root,
            )
            first = creator_cluster_bootstrap(
                validated.evaluated_records, replicates=25, seed=123, label="test"
            )
            second = creator_cluster_bootstrap(
                tuple(reversed(validated.evaluated_records)),
                replicates=25,
                seed=123,
                label="test",
            )
        self.assertEqual(first, second)
        self.assertEqual(first["creator_clusters_↑"], 2)
        self.assertEqual(first["scene_clusters_↑"], 2)
        self.assertFalse(first["row_bootstrap_used"])

    def test_exact_id_coverage_and_forbidden_scoring_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10ScoringFixture(Path(temporary))
            prediction_path = fixture.prediction_root / "prediction_manifest.jsonl"
            rows = [
                json.loads(line) for line in prediction_path.read_text().splitlines()
            ]
            _jsonl(prediction_path, rows[:-1])
            with self.assertRaisesRegex(Real10ScoringError, "coverage is not exact"):
                validate_scoring_inputs(
                    scoring_manifest_path=fixture.scoring_manifest,
                    dataset_root=fixture.dataset_root,
                    prediction_root=fixture.prediction_root,
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10ScoringFixture(Path(temporary))
            rows = [
                json.loads(line)
                for line in fixture.scoring_manifest.read_text().splitlines()
            ]
            rows[0]["sd_sdr"] = 12.0
            _jsonl(fixture.scoring_manifest, rows)
            with self.assertRaisesRegex(Real10ScoringError, "oracle clean-stem"):
                validate_scoring_inputs(
                    scoring_manifest_path=fixture.scoring_manifest,
                    dataset_root=fixture.dataset_root,
                    prediction_root=fixture.prediction_root,
                )

    def test_independent_reconstruction_gate_rejects_self_declared_bad_stems(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10ScoringFixture(Path(temporary), broken_reconstruction=True)
            with self.assertRaisesRegex(Real10ScoringError, r"E\+R-X gate failed"):
                validate_scoring_inputs(
                    scoring_manifest_path=fixture.scoring_manifest,
                    dataset_root=fixture.dataset_root,
                    prediction_root=fixture.prediction_root,
                )

    def test_receipt_and_stem_hash_tampering_fail_before_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10ScoringFixture(Path(temporary))
            receipt_path = fixture.prediction_root / "render_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["run_identity_sha256"] = "0" * 64
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Real10ScoringError, "run identity SHA256"):
                validate_scoring_inputs(
                    scoring_manifest_path=fixture.scoring_manifest,
                    dataset_root=fixture.dataset_root,
                    prediction_root=fixture.prediction_root,
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10ScoringFixture(Path(temporary))
            stem = next((fixture.prediction_root / "stems").rglob("evidence.wav"))
            content = bytearray(stem.read_bytes())
            content[-1] ^= 1
            stem.write_bytes(content)
            with self.assertRaisesRegex(Real10ScoringError, "identity is invalid"):
                validate_scoring_inputs(
                    scoring_manifest_path=fixture.scoring_manifest,
                    dataset_root=fixture.dataset_root,
                    prediction_root=fixture.prediction_root,
                )

    def test_creator_cluster_invariant_is_enforced_from_scoring_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10ScoringFixture(Path(temporary))
            rows = [
                json.loads(line)
                for line in fixture.scoring_manifest.read_text().splitlines()
            ]
            for row in rows:
                if row["scene_id"] == "scene_1":
                    row["creator_id"] = "creator-0"
            _jsonl(fixture.scoring_manifest, rows)
            with self.assertRaisesRegex(Real10ScoringError, "multiple scenes"):
                validate_scoring_inputs(
                    scoring_manifest_path=fixture.scoring_manifest,
                    dataset_root=fixture.dataset_root,
                    prediction_root=fixture.prediction_root,
                )

    def test_real_test_needs_explicit_authorization_and_live_freeze_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10ScoringFixture(Path(temporary), split="real_test")
            with self.assertRaisesRegex(Real10ScoringError, "explicit allow_real_test"):
                validate_scoring_inputs(
                    scoring_manifest_path=fixture.scoring_manifest,
                    dataset_root=fixture.dataset_root,
                    prediction_root=fixture.prediction_root,
                    split="real_test",
                )
            report = score_qces_real10(
                scoring_manifest_path=fixture.scoring_manifest,
                dataset_root=fixture.dataset_root,
                prediction_root=fixture.prediction_root,
                split="real_test",
                allow_real_test=True,
                bootstrap_replicates=5,
            )
            self.assertTrue(
                report["integrity_gates"]["real_test_method_freeze_gate_verified"]
            )

    def test_report_write_is_exclusive_and_has_no_negative_residual_endpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Real10ScoringFixture(root)
            report = score_qces_real10(
                scoring_manifest_path=fixture.scoring_manifest,
                dataset_root=fixture.dataset_root,
                prediction_root=fixture.prediction_root,
                bootstrap_replicates=5,
            )
            output = root / "report.json"
            write_score_report_exclusive(output, report)
            with self.assertRaises(FileExistsError):
                write_score_report_exclusive(output, report)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], REPORT_FORMAT)

        def keys(value: object) -> list[str]:
            if isinstance(value, dict):
                return [str(key) for key in value] + [
                    item for child in value.values() for item in keys(child)
                ]
            if isinstance(value, list):
                return [item for child in value for item in keys(child)]
            return []

        self.assertFalse(any("residual_leakage" in key for key in keys(payload)))

    def test_cli_defaults_to_dev_and_exclusively_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Real10ScoringFixture(root)
            output = root / "cli-report.json"
            arguments = [
                "--scoring-manifest",
                str(fixture.scoring_manifest),
                "--dataset-root",
                str(fixture.dataset_root),
                "--prediction-root",
                str(fixture.prediction_root),
                "--output",
                str(output),
                "--bootstrap-replicates",
                "3",
            ]
            self.assertEqual(scoring_cli_main(arguments), 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["split"], "real_dev"
            )
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                scoring_cli_main(arguments)


if __name__ == "__main__":
    unittest.main()
