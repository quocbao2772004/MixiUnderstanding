"""Tests for the fingerprint-bound semantic reachability diagnostic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mixi_understanding.scripts.audit_qces_semantic_geometry import (
    FOUNDATION_RECEIPT_FORMAT,
    QUESTION_FORMAT,
    SCENE_FORMAT,
    TARGET_FORMAT,
    build_report,
    file_identity,
)


class SemanticGeometryAuditTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        manifest = root / "manifest.jsonl"
        records = [
            {
                "id": "answerable",
                "scene_id": "scene",
                "duration_seconds": 10.0,
                "no_evidence": False,
                "evidence_event_ids": ["event"],
                "events": [
                    {
                        "event_id": "event",
                        "onset_seconds": 1.0,
                        "offset_seconds": 2.0,
                    }
                ],
            },
            {
                "id": "negative",
                "scene_id": "scene",
                "duration_seconds": 10.0,
                "no_evidence": True,
                "evidence_event_ids": [],
                "events": [],
            },
        ]
        manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        manifest_hash = file_identity(manifest)["sha256"]

        cache = root / "foundation"
        cache.mkdir()
        acoustic = torch.zeros(32, 512)
        acoustic[:, 0] = 1.0
        question = torch.zeros(512)
        question[1] = 1.0
        scene_path = cache / "scene_audio_features.pt"
        question_path = cache / "question_features.pt"
        torch.save(
            {
                "format": SCENE_FORMAT,
                "features": {"scene": acoustic},
            },
            scene_path,
        )
        torch.save(
            {
                "format": QUESTION_FORMAT,
                "features": {
                    "answerable": question,
                    "negative": torch.nn.functional.normalize(
                        torch.arange(1, 513, dtype=torch.float32), dim=0
                    ),
                },
            },
            question_path,
        )
        receipt = {
            "format": FOUNDATION_RECEIPT_FORMAT,
            "manifest": {"sha256": manifest_hash},
            "artifacts": {
                "scene_audio_features": {
                    "filename": scene_path.name,
                    "sha256": file_identity(scene_path)["sha256"],
                },
                "question_features": {
                    "filename": question_path.name,
                    "sha256": file_identity(question_path)["sha256"],
                },
            },
        }
        (cache / "cache_receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )

        targets = root / "semantic.pt"
        torch.save(
            {
                "format": TARGET_FORMAT,
                "manifest_sha256": manifest_hash,
                "targets": {"answerable": question},
                "no_evidence_ids": ["negative"],
            },
            targets,
        )
        return manifest, cache, targets

    def test_report_distinguishes_acoustic_and_question_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, cache, targets = self._fixture(Path(directory))
            report = build_report(manifest, cache, targets)
            metrics = report["metrics"]
            self.assertEqual(report["counts"]["answerable_records_↑"], 1)
            self.assertAlmostEqual(
                metrics["oracle_window_scene_clap_target_cosine"]["mean_↑"],
                0.0,
                places=6,
            )
            self.assertAlmostEqual(
                metrics["full_question_clap_target_cosine"]["mean_↑"],
                1.0,
                places=6,
            )
            self.assertGreater(
                metrics[
                    "convex_oracle_candidate_current_initial_target_cosine"
                ]["mean_↑"],
                0.70,
            )

    def test_manifest_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, cache, targets = self._fixture(Path(directory))
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "exact manifest"):
                build_report(manifest, cache, targets)


if __name__ == "__main__":
    unittest.main()
