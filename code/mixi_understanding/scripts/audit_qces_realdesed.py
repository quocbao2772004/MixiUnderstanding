#!/usr/bin/env python3
"""Audit a derived QCES RealDESED set and create a listening queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qces_realdesed import (  # noqa: E402
    NUM_CHANNELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    validate_inference_record,
    validate_scoring_record,
)


AUDIT_SCHEMA_VERSION = "qces_realdesed_audit_v1"
LISTENING_SCHEMA_VERSION = "qces_realdesed_listening_item_v1"


class AuditError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise AuditError(f"{path}:{line_number} is not an object")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _overlap(first: Sequence[float], second: Sequence[float]) -> float:
    return max(0.0, min(float(first[1]), float(second[1])) - max(float(first[0]), float(second[0])))


def _audio_contract(path: Path, expected_sha256: str) -> None:
    if _sha256(path) != expected_sha256:
        raise AuditError(f"canonical audio hash mismatch: {path}")
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != NUM_CHANNELS
            or audio.getsampwidth() != 2
            or audio.getframerate() != SAMPLE_RATE
            or audio.getnframes() != NUM_SAMPLES
            or audio.getcomptype() != "NONE"
        ):
            raise AuditError(f"canonical audio contract mismatch: {path}")


def _risk(event: Mapping[str, Any], overlap_count: int) -> float:
    diagnostic = event["acoustic_diagnostic_not_semantic_proof"]
    rms = float(diagnostic["interval_rms_dbfs_↑"])
    contrast = float(diagnostic["interval_to_context_rms_db_↑"])
    return max(0.0, -35.0 - rms) + max(0.0, -contrast) + 3.0 * overlap_count


def audit(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = dataset_root / "scenes.scoring.jsonl"
    inference_path = dataset_root / "qces_realdesed_inference.jsonl"
    scoring_path = dataset_root / "qces_realdesed_scoring.jsonl"
    build_report_path = dataset_root / "build_report.json"
    scenes = _read_jsonl(scene_path)
    inference = _read_jsonl(inference_path)
    scoring = _read_jsonl(scoring_path)
    report = json.loads(build_report_path.read_text(encoding="utf-8"))
    for row in inference:
        validate_inference_record(row)
    for row in scoring:
        validate_scoring_record(row)
    if report["inference_manifest_sha256"] != _sha256(inference_path):
        raise AuditError("inference manifest hash differs from build report")
    if report["scoring_manifest_sha256"] != _sha256(scoring_path):
        raise AuditError("scoring manifest hash differs from build report")
    if report["scene_manifest_sha256"] != _sha256(scene_path):
        raise AuditError("scene manifest hash differs from build report")

    scene_by_id = {scene["scene_id"]: scene for scene in scenes}
    if len(scene_by_id) != len(scenes):
        raise AuditError("duplicate scene ID")
    inference_by_id = {row["id"]: row for row in inference}
    scoring_by_id = {row["id"]: row for row in scoring}
    if len(inference_by_id) != len(inference) or len(scoring_by_id) != len(scoring):
        raise AuditError("duplicate question ID")
    if set(inference_by_id) != set(scoring_by_id):
        raise AuditError("inference/scoring ID set mismatch")

    event_candidates = []
    class_counts = Counter()
    overlap_events = 0
    for scene in scenes:
        audio_path = (dataset_root / scene["canonical_audio_path"]).resolve()
        try:
            audio_path.relative_to(dataset_root)
        except ValueError as error:
            raise AuditError("canonical path escapes dataset root") from error
        _audio_contract(audio_path, scene["canonical_audio_sha256"])
        events = scene["events"]
        event_by_id = {event["event_id"]: event for event in events}
        if len(event_by_id) != len(events):
            raise AuditError(f"{scene['scene_id']}: duplicate event ID")
        for event in events:
            relative = event["relative_interval"]
            absolute = event["absolute_interval"]
            source_annotation = event["source_annotation_interval"]
            if not 0 <= relative[0] < relative[1] <= 10:
                raise AuditError(f"{scene['scene_id']}: event outside canonical crop")
            shift = float(scene["crop_start_seconds"])
            if not (
                math.isclose(relative[0] + shift, absolute[0], abs_tol=2e-6)
                and math.isclose(relative[1] + shift, absolute[1], abs_tol=2e-6)
            ):
                raise AuditError(f"{scene['scene_id']}: absolute/relative timestamp drift")
            if not math.isclose(source_annotation[0], absolute[0], abs_tol=2e-6):
                raise AuditError(f"{scene['scene_id']}: annotation onset was censored")
            right_censored = bool(event["right_censored_by_crop"])
            if right_censored != (source_annotation[1] > absolute[1] + 2e-6):
                raise AuditError(f"{scene['scene_id']}: inconsistent censor flag")
            if source_annotation[1] + 2e-6 < absolute[1]:
                raise AuditError(f"{scene['scene_id']}: visible event exceeds annotation")
            overlapping = [
                other["event_id"]
                for other in events
                if other["event_id"] != event["event_id"]
                and _overlap(relative, other["relative_interval"]) > 0
            ]
            if overlapping:
                overlap_events += 1
            class_counts[event["class"]] += 1
            event_candidates.append(
                {
                    "schema_version": LISTENING_SCHEMA_VERSION,
                    "scene_id": scene["scene_id"],
                    "event_id": event["event_id"],
                    "class": event["class"],
                    "display_name": event["display_name"],
                    "canonical_audio_path": scene["canonical_audio_path"],
                    "canonical_interval": relative,
                    "source_absolute_interval": absolute,
                    "source_annotation_interval": source_annotation,
                    "right_censored_by_crop": right_censored,
                    "overlapping_event_ids": overlapping,
                    "acoustic_diagnostic_not_semantic_proof": event[
                        "acoustic_diagnostic_not_semantic_proof"
                    ],
                    "risk_score_↓": round(_risk(event, len(overlapping)), 4),
                    "listen_protocol": (
                        "listen to full 10 s once, then boosted interval, then "
                        "true-level interval; mark accept/reject/uncertain"
                    ),
                    "required_human_fields": {
                        "decision": None,
                        "label_audible": None,
                        "timestamp_acceptable": None,
                        "notes": None,
                    },
                }
            )

        for row in (item for item in scoring if item["scene_id"] == scene["scene_id"]):
            for role in ("anchor", "answer", "evidence"):
                ids = row[f"{role}_event_ids"]
                intervals = row[f"{role}_intervals"]
                for event_id, interval in zip(ids, intervals):
                    if event_id not in event_by_id:
                        raise AuditError(f"{row['id']}: scoring references unknown event")
                    if interval != event_by_id[event_id]["relative_interval"]:
                        raise AuditError(f"{row['id']}: scoring/scene interval mismatch")

    # Two deliberately difficult examples per represented class.  This makes
    # the human queue compact while covering every class and prioritizing low
    # level/low contrast/overlap cases.
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in event_candidates:
        by_class[candidate["class"]].append(candidate)
    queue = []
    for label in sorted(by_class):
        ranked = sorted(
            by_class[label],
            key=lambda item: (-float(item["risk_score_↓"]), item["scene_id"], item["event_id"]),
        )
        queue.extend(ranked[:2])
    queue = sorted(
        queue,
        key=lambda item: (-float(item["risk_score_↓"]), item["class"], item["scene_id"]),
    )
    _write_jsonl(output_dir / "manual_listening_queue.jsonl", queue)

    relation_counts = Counter(row["relation"] for row in scoring)
    option_positions = Counter(row["answer_option_index"] for row in scoring)
    audit_report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "build_contract_sha256": report["dataset_contract_sha256"],
        "integrity": {
            "canonical_audio_hash_failures_↓": 0,
            "canonical_audio_spec_failures_↓": 0,
            "manifest_binding_failures_↓": 0,
            "scoring_scene_timestamp_mismatches_↓": 0,
            "unknown_event_references_↓": 0,
        },
        "counts": {
            "scenes_↑": len(scenes),
            "events_↑": len(event_candidates),
            "questions_↑": len(scoring),
            "answerable_questions_↑": sum(not row["no_evidence"] for row in scoring),
            "no_evidence_questions_↑": sum(row["no_evidence"] for row in scoring),
            "events_with_overlap_↑": overlap_events,
            "represented_classes_↑": len(class_counts),
            "events_by_class": dict(sorted(class_counts.items())),
            "questions_by_relation": dict(sorted(relation_counts.items())),
            "answer_option_position_counts": {
                str(key): value for key, value in sorted(option_positions.items())
            },
            "manual_listening_items_↓": len(queue),
        },
        "human_gate": {
            "status": "pending",
            "acceptance_rule": (
                "every queued item must be marked accept by a human; reject or "
                "uncertain items are excluded by immutable scene/event ID"
            ),
            "queue": str((output_dir / "manual_listening_queue.jsonl").resolve()),
        },
        "metric_directions": {
            "temporal_iou": "↑",
            "qa_sufficiency": "↑",
            "residual_answer_leakage": "↓",
            "retained_duration": "↓",
            "no_evidence_auroc": "↑",
            "waveform_sdr": "FORBIDDEN",
        },
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(audit_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checklist = [
        "# QCES RealDESED manual listening checklist",
        "",
        "Nghe full 10 s trước, rồi nghe boosted interval và true-level interval trên",
        "tab `Real audio data`. Không dùng boosted audio làm model input hoặc metric.",
        "",
        f"Queue có **{len(queue)}** event: tối đa 2 case khó cho mỗi class.",
        "",
    ]
    for index, item in enumerate(queue, 1):
        onset, offset = item["canonical_interval"]
        checklist.append(
            f"- [ ] {index:02d}. `{item['scene_id']}` / `{item['event_id']}` · "
            f"**{item['display_name']}** · {onset:.2f}–{offset:.2f}s · "
            f"risk {item['risk_score_↓']:.2f} ↓"
        )
    (output_dir / "LISTENING_CHECKLIST.md").write_text(
        "\n".join(checklist) + "\n", encoding="utf-8"
    )
    return audit_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = audit(args.dataset_root, args.output_dir)
    except (AuditError, OSError, ValueError, wave.Error) as error:
        raise SystemExit(f"QCES RealDESED audit failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
