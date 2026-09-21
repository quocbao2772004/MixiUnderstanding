#!/usr/bin/env python3
"""Paired BEATs curator-consistency audit for old/new QCES semantic crops.

This is deliberately not an independent semantic evaluation: the same frozen
BEATs model selected the crop bank. Its only purpose is to catch rendering
mistakes between a crop-bank center and the actual variable-duration stems.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf
import torch

from mixi_understanding.qces.beats_auditor import BEATsAudioSetAuditor


FORMAT = "qces_v5_semantic_crop_smoke_audit_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    baseline = PROJECT_ROOT / "code/baseline/beats-unilm"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beats-root", type=Path, default=baseline)
    parser.add_argument(
        "--beats-checkpoint",
        type=Path,
        default=baseline
        / "checkpoint/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    )
    parser.add_argument(
        "--audioset-labels",
        type=Path,
        default=baseline / "checkpoint/class_labels_indices.csv",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_report(path: Path) -> tuple[Mapping[str, Any], dict[tuple[str, str], Mapping[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != (
        "qces_v5_acoustic_audibility_audit_v1"
    ):
        raise ValueError(f"invalid audibility report: {path}")
    result = {}
    for event in payload.get("events", []):
        if not isinstance(event, dict) or event.get("event_kind") != "semantic":
            continue
        key = (str(event["scene_id"]), str(event["event_id"]))
        if key in result:
            raise ValueError(f"duplicate semantic event: {key}")
        result[key] = event
    if not result:
        raise ValueError(f"audibility report has no semantic events: {path}")
    return payload, result


def read_stem(report: Mapping[str, Any], event: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    manifest = report.get("manifest")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("path"), str):
        raise ValueError("report lacks a bound manifest")
    root = Path(str(manifest["path"])).resolve().parent
    relative = event.get("stem_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("event lacks stem_path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"stem escapes artifact root: {relative}") from error
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1, dtype=np.float64).astype(np.float32)
    if sample_rate != 32_000 or mono.shape != (320_000,) or not np.isfinite(mono).all():
        raise ValueError(f"invalid stem audio: {path}")
    return mono, sample_rate


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reference_path = args.reference_report.expanduser().resolve()
    candidate_path = args.candidate_report.expanduser().resolve()
    reference_report, reference_events = load_report(reference_path)
    candidate_report, candidate_events = load_report(candidate_path)
    if not set(candidate_events).issubset(reference_events):
        raise ValueError(
            "candidate semantic event keys are not contained in the reference report"
        )
    labels = {str(event["label"]) for event in candidate_events.values()}
    for key in candidate_events:
        if reference_events[key].get("label") != candidate_events[key].get("label"):
            raise ValueError(f"semantic label changed for {key}")
    auditor = BEATsAudioSetAuditor.from_assets(
        args.beats_root,
        args.beats_checkpoint,
        args.audioset_labels,
        labels,
        torch.device(args.device),
    )
    keys = sorted(candidate_events)
    items = []
    for start in range(0, len(keys), args.batch_size):
        chunk = keys[start : start + args.batch_size]
        waveforms = []
        for key in chunk:
            waveforms.extend(
                (
                    read_stem(reference_report, reference_events[key])[0],
                    read_stem(candidate_report, candidate_events[key])[0],
                )
            )
        probabilities = auditor.score(
            torch.from_numpy(np.stack(waveforms)), 32_000, args.batch_size * 2
        )
        for index, key in enumerate(chunk):
            label = str(candidate_events[key]["label"])
            coordinate = auditor.label_indices[label]
            reference_probability = float(probabilities[index * 2, coordinate])
            candidate_probability = float(probabilities[index * 2 + 1, coordinate])
            delta = candidate_probability - reference_probability
            items.append(
                {
                    "scene_id": key[0],
                    "event_id": key[1],
                    "label": label,
                    "source_id": candidate_events[key].get("source_id"),
                    "reference_crop_interval_seconds": reference_events[key].get(
                        "source_crop_interval_seconds"
                    ),
                    "candidate_crop_interval_seconds": candidate_events[key].get(
                        "source_crop_interval_seconds"
                    ),
                    "reference_label_probability ↑": reference_probability,
                    "candidate_label_probability ↑": candidate_probability,
                    "candidate_minus_reference ↑": delta,
                    "degraded_by_at_least_0.05 ↓": delta <= -0.05,
                    "catastrophic_degradation ↓": (
                        delta <= -0.05 and candidate_probability < 0.05
                    ),
                }
            )
    reference = np.asarray(
        [item["reference_label_probability ↑"] for item in items], dtype=np.float64
    )
    candidate = np.asarray(
        [item["candidate_label_probability ↑"] for item in items], dtype=np.float64
    )
    delta = candidate - reference
    catastrophic_rate = float(
        np.mean([bool(item["catastrophic_degradation ↓"]) for item in items])
    )
    metrics = {
        "paired_semantic_events ↑": len(items),
        "reference_mean_label_probability ↑": float(reference.mean()),
        "candidate_mean_label_probability ↑": float(candidate.mean()),
        "mean_delta ↑": float(delta.mean()),
        "reference_p05_label_probability ↑": float(np.quantile(reference, 0.05)),
        "candidate_p05_label_probability ↑": float(np.quantile(candidate, 0.05)),
        "p05_delta ↑": float(
            np.quantile(candidate, 0.05) - np.quantile(reference, 0.05)
        ),
        "improved_event_rate ↑": float(np.mean(delta > 0.0)),
        "degraded_by_at_least_0.05_rate ↓": float(np.mean(delta <= -0.05)),
        "catastrophic_degradation_rate ↓": catastrophic_rate,
    }
    gates = {
        "mean_delta_positive": metrics["mean_delta ↑"] > 0.0,
        "p05_delta_positive": metrics["p05_delta ↑"] > 0.0,
        "catastrophic_degradation_rate_at_most_0.05": catastrophic_rate <= 0.05,
    }
    return {
        "format": FORMAT,
        "claim_boundary": (
            "BEATs selected the crop bank, so this only checks curator/render "
            "consistency and is forbidden as independent semantic evaluation."
        ),
        "reference_report": identity(reference_path),
        "candidate_report": identity(candidate_path),
        "beats_provenance": auditor.provenance,
        "metrics": metrics,
        "gates": gates,
        "automated_curator_consistency_gate_passed": all(gates.values()),
        "human_semantic_gate_still_required": True,
        "items": items,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"gate passed: {report['automated_curator_consistency_gate_passed']}")
    print(f"report ready: {output}")


if __name__ == "__main__":
    main()
