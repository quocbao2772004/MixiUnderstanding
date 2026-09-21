#!/usr/bin/env python3
"""Audit whether QCES semantic mixing can reach AudioSep supervision targets.

This is a training-supervision diagnostic, never an evaluation baseline.  It
binds the exact manifest, frozen AudioSep-CLAP cache, and semantic-target cache;
then compares oracle evidence-window acoustic CLAP, full-question CLAP, and the
two checkpointed mixer parameterizations in their shared 512-D space.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F

from mixi_understanding.qces.composer import role_pool_scene_clap


FORMAT_VERSION = "qces_semantic_geometry_audit_v2"
FOUNDATION_RECEIPT_FORMAT = "qces_audiosep_clap_frozen_controller_cache_receipt_v1"
SCENE_FORMAT = "qces_audiosep_clap_scene_features_v1"
QUESTION_FORMAT = "qces_audiosep_clap_question_features_v1"
TARGET_FORMAT = "qces_audiosep_union_semantic_targets_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--foundation-cache", type=Path, required=True)
    parser.add_argument("--semantic-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_manifest(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError("manifest is empty")
    ids = [record.get("id") for record in records]
    if any(not isinstance(sample_id, str) for sample_id in ids):
        raise ValueError("every manifest record needs a string id")
    if len(set(ids)) != len(ids):
        raise ValueError("manifest sample IDs are not unique")
    return records


def _require_tensor_map(
    payload: Mapping[str, Any], key: str, expected_ids: set[str], shape: tuple[int, ...]
) -> Mapping[str, torch.Tensor]:
    values = payload.get(key)
    if not isinstance(values, dict) or set(values) != expected_ids:
        raise ValueError(f"{key} IDs differ from the manifest contract")
    for item_id, value in values.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"invalid {key} tensor for {item_id}")
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite {key} tensor for {item_id}")
    return values


def _oracle_union_mask(record: Mapping[str, Any], frames: int = 1001) -> torch.Tensor:
    duration = float(record["duration_seconds"])
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"invalid duration for {record['id']}")
    evidence_ids = set(record.get("evidence_event_ids", []))
    if not evidence_ids:
        raise ValueError(f"answerable record has no evidence IDs: {record['id']}")
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError(f"record has no event list: {record['id']}")
    timeline = torch.linspace(0.0, duration, frames)
    mask = torch.zeros(frames)
    matched: set[str] = set()
    for event in events:
        if not isinstance(event, dict) or event.get("event_id") not in evidence_ids:
            continue
        event_id = str(event["event_id"])
        onset = float(event["onset_seconds"])
        offset = float(event["offset_seconds"])
        if not 0.0 <= onset < offset <= duration + 1e-6:
            raise ValueError(f"invalid evidence interval for {record['id']}:{event_id}")
        mask = torch.maximum(
            mask, ((timeline >= onset) & (timeline <= offset)).to(mask.dtype)
        )
        matched.add(event_id)
    if matched != evidence_ids or not mask.any():
        raise ValueError(f"evidence IDs/intervals are incomplete for {record['id']}")
    return mask


def _stats(values: torch.Tensor) -> dict[str, float]:
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("metric vector must be finite and non-empty")
    return {
        "mean_↑": float(values.mean()),
        "median_↑": float(values.median()),
        "minimum_↑": float(values.min()),
        "maximum_↑": float(values.max()),
    }


def build_report(
    manifest: Path, foundation_cache: Path, semantic_targets: Path
) -> dict[str, Any]:
    manifest_identity = file_identity(manifest)
    records = _load_manifest(manifest)
    sample_ids = {str(record["id"]) for record in records}
    answerable = [record for record in records if not bool(record.get("no_evidence"))]
    answerable_ids = {str(record["id"]) for record in answerable}
    scene_ids = {str(record["scene_id"]) for record in records}

    receipt_path = foundation_cache / "cache_receipt.json"
    receipt = _load_json(receipt_path)
    if receipt.get("format") != FOUNDATION_RECEIPT_FORMAT:
        raise ValueError("unsupported foundation cache receipt format")
    declared_manifest = receipt.get("manifest")
    if not isinstance(declared_manifest, dict) or declared_manifest.get(
        "sha256"
    ) != manifest_identity["sha256"]:
        raise ValueError("foundation cache is not bound to the exact manifest")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("foundation receipt has no artifact identities")

    def load_foundation_artifact(name: str) -> Mapping[str, Any]:
        declared = artifacts.get(name)
        if not isinstance(declared, dict) or not isinstance(
            declared.get("filename"), str
        ):
            raise ValueError(f"foundation receipt is missing {name}")
        path = foundation_cache / declared["filename"]
        identity = file_identity(path)
        if identity["sha256"] != declared.get("sha256"):
            raise ValueError(f"foundation artifact hash mismatch: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid foundation artifact: {name}")
        return payload

    scene_payload = load_foundation_artifact("scene_audio_features")
    question_payload = load_foundation_artifact("question_features")
    if scene_payload.get("format") != SCENE_FORMAT:
        raise ValueError("unsupported scene-feature cache format")
    if question_payload.get("format") != QUESTION_FORMAT:
        raise ValueError("unsupported question-feature cache format")
    scene_features = _require_tensor_map(
        scene_payload, "features", scene_ids, (32, 512)
    )
    question_features = _require_tensor_map(
        question_payload, "features", sample_ids, (512,)
    )

    target_identity = file_identity(semantic_targets)
    target_payload = torch.load(
        semantic_targets, map_location="cpu", weights_only=False
    )
    if not isinstance(target_payload, dict) or target_payload.get("format") != TARGET_FORMAT:
        raise ValueError("unsupported semantic-target cache format")
    if target_payload.get("manifest_sha256") != manifest_identity["sha256"]:
        raise ValueError("semantic targets are not bound to the exact manifest")
    targets = _require_tensor_map(
        target_payload, "targets", answerable_ids, (512,)
    )
    if set(target_payload.get("no_evidence_ids", [])) != sample_ids - answerable_ids:
        raise ValueError("semantic-target no-evidence partition is invalid")

    rows = []
    legacy_gate = float(torch.tensor(-2.0).sigmoid())
    convex_gate = 0.5
    for record in answerable:
        sample_id = str(record["id"])
        mask = _oracle_union_mask(record)
        acoustic = role_pool_scene_clap(
            scene_features[str(record["scene_id"])].float().unsqueeze(0),
            mask.unsqueeze(0),
        )[0]
        full_scene_mean = F.normalize(
            scene_features[str(record["scene_id"])].float().mean(dim=0), dim=0
        )
        question = F.normalize(question_features[sample_id].float(), dim=0)
        target = F.normalize(targets[sample_id].float(), dim=0)
        legacy_oracle_candidate = F.normalize(
            acoustic + legacy_gate * target, dim=0
        )
        convex_old_gate_oracle_candidate = F.normalize(
            (1.0 - legacy_gate) * acoustic + legacy_gate * target, dim=0
        )
        convex_current_oracle_candidate = F.normalize(
            (1.0 - convex_gate) * acoustic + convex_gate * target, dim=0
        )
        rows.append(
            torch.stack(
                [
                    acoustic @ target,
                    question @ target,
                    legacy_oracle_candidate @ target,
                    convex_old_gate_oracle_candidate @ target,
                    convex_current_oracle_candidate @ target,
                    full_scene_mean @ target,
                    full_scene_mean @ question,
                ]
            )
        )
    matrix = torch.stack(rows)
    metrics = {
        "oracle_window_scene_clap_target_cosine": _stats(matrix[:, 0]),
        "full_question_clap_target_cosine": _stats(matrix[:, 1]),
        "legacy_oracle_candidate_initial_target_cosine": _stats(matrix[:, 2]),
        "convex_oracle_candidate_gate_0.119_target_cosine": _stats(matrix[:, 3]),
        "convex_oracle_candidate_current_initial_target_cosine": _stats(matrix[:, 4]),
        "full_scene_mean_fine_clap_target_cosine": _stats(matrix[:, 5]),
        "full_scene_mean_fine_clap_question_cosine": _stats(matrix[:, 6]),
        # The question-residual adapter has an exactly zero final layer, so this
        # is its actual (not oracle-candidate) semantic condition at step zero.
        "question_residual_actual_initial_target_cosine": _stats(matrix[:, 1]),
    }
    return {
        "format": FORMAT_VERSION,
        "scope": "training_supervision_geometry_not_evaluation_result",
        "inputs": {
            "manifest": manifest_identity,
            "foundation_receipt": file_identity(receipt_path),
            "semantic_targets": target_identity,
        },
        "counts": {
            "records_↑": len(records),
            "answerable_records_↑": len(answerable),
            "scene_ids_↑": len(scene_ids),
        },
        "mixer_contract": {
            "legacy_initial_candidate_weight_descriptive": legacy_gate,
            "convex_current_initial_candidate_weight_descriptive": convex_gate,
            "legacy_initial_maximum_angular_shift_degrees_descriptive": math.degrees(
                math.asin(legacy_gate)
            ),
            "oracle_candidate_warning": (
                "Mixer rows substitute the normalized semantic target as the learned "
                "candidate to measure parameterization geometry; they are not an "
                "untrained-model performance measurement."
            ),
        },
        "metrics": metrics,
        "conclusion": {
            "acoustic_base_is_near_orthogonal": abs(
                metrics["oracle_window_scene_clap_target_cosine"]["mean_↑"]
            )
            < 0.1,
            "question_embedding_is_materially_aligned": metrics[
                "full_question_clap_target_cosine"
            ]["mean_↑"]
            > 0.3,
            "fine_frame_projection_is_not_a_calibrated_semantic_base": abs(
                metrics["full_scene_mean_fine_clap_target_cosine"]["mean_↑"]
            )
            < 0.1,
            "reachable_convex_endpoint_required": True,
            "question_residual_identity_prior_preferred_for_next_microfit": True,
        },
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }


def _atomic_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved.exists() and not overwrite:
        raise ValueError(f"output exists; pass --overwrite: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, resolved)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(
        args.manifest.resolve(),
        args.foundation_cache.resolve(),
        args.semantic_targets.resolve(),
    )
    _atomic_json(args.output, report, args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
