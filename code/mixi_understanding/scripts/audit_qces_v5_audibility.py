#!/usr/bin/env python3
"""Audit acoustic audibility proxies for every rendered QCES-v5 event.

This is a dataset-quality screen, not a proof that a human recognizes an
event.  It measures each rendered stem over its annotated interval against
the rest of the mixture, then prioritizes weak anchor/answer events for a
semantic-model and human listening audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf


FORMAT_VERSION = "qces_v5_acoustic_audibility_audit_v1"
DB_FLOOR = -120.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
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


def load_manifest(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
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
    return records


def safe_resolve(root: Path, relative: object, context: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{context} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes manifest root: {relative}") from error
    if not candidate.is_file():
        raise ValueError(f"missing {context}: {candidate}")
    return candidate


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.size == 0 or sample_rate <= 0:
        raise ValueError(f"empty or invalid audio: {path}")
    mono = waveform.mean(axis=1, dtype=np.float64)
    if not np.isfinite(mono).all():
        raise ValueError(f"non-finite audio: {path}")
    return mono, int(sample_rate)


def power_db(signal: np.ndarray) -> float:
    if signal.size == 0:
        return DB_FLOOR
    power = float(np.mean(np.square(signal, dtype=np.float64)))
    if power <= 1e-12:
        return DB_FLOOR
    return max(DB_FLOOR, 10.0 * math.log10(power))


def ratio_db(numerator: np.ndarray, denominator: np.ndarray) -> float:
    numerator_power = float(np.mean(np.square(numerator, dtype=np.float64)))
    denominator_power = float(np.mean(np.square(denominator, dtype=np.float64)))
    return float(
        np.clip(
            10.0 * math.log10((numerator_power + 1e-12) / (denominator_power + 1e-12)),
            DB_FLOOR,
            -DB_FLOOR,
        )
    )


def summary(values: Sequence[float], *, direction: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("summary requires finite non-empty values")
    return {
        f"mean {direction}": float(array.mean()),
        f"median {direction}": float(np.median(array)),
        f"p05 {direction}": float(np.quantile(array, 0.05)),
        f"p25 {direction}": float(np.quantile(array, 0.25)),
        f"p75 {direction}": float(np.quantile(array, 0.75)),
        f"p95 {direction}": float(np.quantile(array, 0.95)),
        f"minimum {direction}": float(array.min()),
        f"maximum {direction}": float(array.max()),
    }


def screening_band(interval_rms_dbfs: float, snr_db: float) -> str:
    # Diagnostic triage only.  These thresholds are intentionally not called
    # an audibility ground truth because recognition also depends on spectrum,
    # event salience, source-label quality, and the listener.
    if interval_rms_dbfs < -45.0 or snr_db < -5.0:
        return "high_masking_or_low_level_risk"
    if interval_rms_dbfs < -35.0 or snr_db < 0.0:
        return "manual_review"
    return "lower_risk_candidate"


def build_report(manifest: Path) -> dict[str, Any]:
    manifest = manifest.resolve()
    root = manifest.parent
    records = load_manifest(manifest)

    scene_rows: dict[str, Mapping[str, Any]] = {}
    scene_recipe_fingerprints: dict[str, str] = {}
    event_roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    event_question_counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        scene_id = record.get("scene_id")
        if not isinstance(scene_id, str):
            raise ValueError("every record needs a string scene_id")
        recipe = {
            "mixture_path": record.get("mixture_path"),
            "sample_rate": record.get("sample_rate"),
            "num_samples": record.get("num_samples"),
            "events": record.get("events"),
        }
        fingerprint = hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        previous = scene_recipe_fingerprints.setdefault(scene_id, fingerprint)
        if previous != fingerprint:
            raise ValueError(f"scene recipe differs across questions: {scene_id}")
        scene_rows.setdefault(scene_id, record)
        for role, field in (
            ("anchor", "anchor_event_ids"),
            ("answer", "answer_event_ids"),
            ("evidence", "evidence_event_ids"),
        ):
            values = record.get(field, [])
            if not isinstance(values, list):
                raise ValueError(f"{field} must be a list for {record.get('id')}")
            for event_id in values:
                if not isinstance(event_id, str):
                    raise ValueError(f"non-string ID in {field}")
                event_roles[(scene_id, event_id)].add(role)
                event_question_counts[(scene_id, event_id)] += 1

    event_results: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    for scene_index, scene_id in enumerate(sorted(scene_rows), 1):
        record = scene_rows[scene_id]
        mixture_path = safe_resolve(root, record.get("mixture_path"), "mixture_path")
        mixture, sample_rate = read_mono(mixture_path)
        expected_rate = int(record.get("sample_rate", -1))
        expected_samples = int(record.get("num_samples", -1))
        if sample_rate != expected_rate or len(mixture) != expected_samples:
            raise ValueError(f"mixture shape/rate mismatch: {scene_id}")
        events = record.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"scene has no events: {scene_id}")
        stem_sum = np.zeros_like(mixture, dtype=np.float64)
        scene_event_results: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                raise ValueError(f"non-object event: {scene_id}")
            event_id = event.get("event_id")
            if not isinstance(event_id, str):
                raise ValueError(f"event without string ID: {scene_id}")
            stem_path = safe_resolve(root, event.get("stem_path"), "stem_path")
            stem, stem_rate = read_mono(stem_path)
            if stem_rate != sample_rate or len(stem) != len(mixture):
                raise ValueError(f"stem shape/rate mismatch: {scene_id}:{event_id}")
            stem_sum += stem
            start = max(0, int(math.floor(float(event["onset_seconds"]) * sample_rate)))
            stop = min(
                len(stem), int(math.ceil(float(event["offset_seconds"]) * sample_rate))
            )
            if stop <= start:
                raise ValueError(f"empty rendered interval: {scene_id}:{event_id}")
            active_stem = stem[start:stop]
            active_mix = mixture[start:stop]
            active_background = active_mix - active_stem
            interval_rms_dbfs = power_db(active_stem)
            event_background_snr_db = ratio_db(active_stem, active_background)
            event_mix_energy_db = ratio_db(active_stem, active_mix)
            peak_dbfs = (
                DB_FLOOR
                if not np.any(active_stem)
                else max(DB_FLOOR, 20.0 * math.log10(float(np.max(np.abs(active_stem)))))
            )
            activity_threshold = max(float(np.max(np.abs(active_stem))) * 0.01, 1e-4)
            activity_fraction = float(np.mean(np.abs(active_stem) >= activity_threshold))
            roles = sorted(event_roles.get((scene_id, event_id), set()))
            result = {
                "split": record.get("split"),
                "scene_id": scene_id,
                "scene_family_id": record.get("scene_family_id"),
                "variant_id": record.get("variant_id"),
                "mixture_path": str(record["mixture_path"]),
                "event_id": event_id,
                "label": event.get("label"),
                "occurrence_index": event.get("occurrence_index"),
                "event_kind": event.get("event_kind"),
                "onset_seconds": float(event["onset_seconds"]),
                "offset_seconds": float(event["offset_seconds"]),
                "stem_path": str(event["stem_path"]),
                "source_id": event.get("source_id"),
                "source_path": event.get("source_path"),
                "source_interval_seconds": event.get("source_interval_seconds"),
                "source_crop_interval_seconds": event.get("source_crop_interval_seconds"),
                "attribution": event.get("attribution"),
                "recipe_gain_db": float(event["gain_db"]),
                "roles_across_scene_questions": roles,
                "role_question_reference_count": event_question_counts[(scene_id, event_id)],
                "required_for_qa": bool(set(roles) & {"anchor", "answer"}),
                "interval_stem_rms_dbfs ↑": interval_rms_dbfs,
                "interval_stem_peak_dbfs ↑": peak_dbfs,
                "event_to_background_snr_db ↑": event_background_snr_db,
                "event_to_mixture_energy_db ↑": event_mix_energy_db,
                "within_interval_activity_fraction ↑": activity_fraction,
                "screening_band": screening_band(
                    interval_rms_dbfs, event_background_snr_db
                ),
            }
            scene_event_results.append(result)
        reconstruction_errors.append(
            float(np.mean(np.abs(mixture.astype(np.float64) - stem_sum)))
        )
        event_results.extend(scene_event_results)
        if scene_index % 100 == 0:
            print(
                f"audited {scene_index}/{len(scene_rows)} scenes",
                flush=True,
            )

    required = [result for result in event_results if result["required_for_qa"]]
    semantic = [
        result for result in event_results if result["event_kind"] == "semantic"
    ]
    bands_all = Counter(result["screening_band"] for result in event_results)
    bands_required = Counter(result["screening_band"] for result in required)

    def metric_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    high_risk_required = [
        result
        for result in required
        if result["screening_band"] == "high_masking_or_low_level_risk"
    ]
    review_required = [
        result
        for result in required
        if result["screening_band"] != "lower_risk_candidate"
    ]
    return {
        "format": FORMAT_VERSION,
        "interpretation": {
            "scope": (
                "Acoustic screening proxy only; it does not prove human audibility "
                "or semantic label correctness."
            ),
            "high_masking_or_low_level_risk": (
                "interval stem RMS < -45 dBFS OR event/background SNR < -5 dB"
            ),
            "manual_review": (
                "not high risk, but interval stem RMS < -35 dBFS OR "
                "event/background SNR < 0 dB"
            ),
            "lower_risk_candidate": (
                "passes the two acoustic screening thresholds; semantic/human "
                "verification is still required"
            ),
        },
        "manifest": file_identity(manifest),
        "counts": {
            "question_record_count": len(records),
            "unique_scene_count": len(scene_rows),
            "unique_event_count": len(event_results),
            "semantic_event_count": len(semantic),
            "qa_required_unique_event_count": len(required),
        },
        "screening_bands_all_events": dict(sorted(bands_all.items())),
        "screening_bands_qa_required_events": dict(sorted(bands_required.items())),
        "aggregate_metrics": {
            "all_event_to_background_snr_db ↑": summary(
                metric_values(event_results, "event_to_background_snr_db ↑"),
                direction="↑",
            ),
            "semantic_event_to_background_snr_db ↑": summary(
                metric_values(semantic, "event_to_background_snr_db ↑"),
                direction="↑",
            ),
            "qa_required_event_to_background_snr_db ↑": summary(
                metric_values(required, "event_to_background_snr_db ↑"),
                direction="↑",
            ),
            "qa_required_interval_stem_rms_dbfs ↑": summary(
                metric_values(required, "interval_stem_rms_dbfs ↑"),
                direction="↑",
            ),
            "qa_required_high_risk_rate ↓": len(high_risk_required) / len(required),
            "qa_required_review_or_high_risk_rate ↓": len(review_required)
            / len(required),
            "mixture_stem_reconstruction_mae ↓": summary(
                reconstruction_errors, direction="↓"
            ),
        },
        "worst_qa_required_events_by_snr": sorted(
            required, key=lambda result: float(result["event_to_background_snr_db ↑"])
        )[:100],
        "events": event_results,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise ValueError(f"output exists; pass --overwrite: {output}")
    report = build_report(args.manifest.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps({"status": "ok", "output": str(output), **report["counts"]}))


if __name__ == "__main__":
    main()
