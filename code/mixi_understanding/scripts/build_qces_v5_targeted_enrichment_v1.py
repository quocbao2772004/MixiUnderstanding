#!/usr/bin/env python3
"""Build a leakage-safe Gold-only enrichment tier for weak V5 classes.

The class list is taken from the frozen semantic-sufficiency audit.  Individual
sources are then selected *only* by source quality from the official AudioSet
train partition.  Existing V5 train/dev and locked-test hard identities are
excluded before selection.  The fixed V5 dev and test manifests are never
rewritten by this builder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.clean_evidence_scenes import (  # noqa: E402
    CleanSource,
    _atomic_write_text,
    load_source_bank,
    sha256_file,
)
from mixi_understanding.scripts import build_qces_tiered_realistic_v5 as v5  # noqa: E402


FORMAT = "qces_v5_targeted_enrichment_v1"
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    / "v5_semantic_sufficiency_audit_v1/report.json"
)
DEFAULT_V5 = PROJECT_ROOT / "outputs/qces_full188_tiered_realistic_v5"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean"
DEFAULT_OUTPUT = Path("/var/tmp/qces_v5_targeted_enrichment_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--v5-root", type=Path, default=DEFAULT_V5)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources-per-class", type=int, default=20)
    parser.add_argument("--minimum-sources-per-class", type=int, default=16)
    parser.add_argument("--scene-count", type=int, default=320)
    parser.add_argument("--seed", type=int, default=6901)
    parser.add_argument("--layout-trials", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _event_identities(event: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (name, str(value))
        for name, value in (
            ("source_id", event.get("source_id")),
            ("video_id", event.get("source_video_id") or event.get("video_id")),
            ("source_sha256", event.get("source_sha256")),
            ("audio_path", event.get("source_path") or event.get("audio_path")),
        )
        if value
    }


def _row_identities(row: Mapping[str, Any], audio_path: Path) -> set[tuple[str, str]]:
    return {
        (name, str(value))
        for name, value in (
            ("source_id", row.get("source_id") or row.get("item_id")),
            ("video_id", row.get("source_video_id") or row.get("video_id")),
            ("source_sha256", row.get("source_sha256") or row.get("stem_sha256")),
            ("audio_path", audio_path.resolve().as_posix()),
        )
        if value
    }


def _resolve_audio_path(row: Mapping[str, Any]) -> Path:
    value = row.get("audio_path") or row.get("stem_path") or row.get("source_path")
    if not value:
        raise ValueError("source-bank row has no audio path")
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _quality_key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
    metrics = row.get("quality_metrics") or {}
    audibility = float(row.get("audibility_score") or metrics.get("target_text_similarity") or -1.0)
    margin = float(metrics.get("target_residual_margin") or -1.0)
    duration = float(row.get("active_offset_seconds", 0.0)) - float(
        row.get("active_onset_seconds", 0.0)
    )
    # The final stable source-id tie break prevents filesystem-order drift.
    source_id = str(row.get("source_id") or row.get("item_id") or "")
    return audibility, margin, duration, source_id


def _existing_identities(v5_root: Path) -> tuple[set[tuple[str, str]], dict[str, int]]:
    manifests = {
        "train": v5_root / "detector_scene_manifest_tiered_train.jsonl",
        "dev": v5_root / "detector_scene_manifest_tiered_dev.jsonl",
        "locked_test": v5_root / "detector_scene_manifest_locked_test_filtered.jsonl",
    }
    identities: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for split, path in manifests.items():
        rows = _read_jsonl(path)
        counts[split] = len(rows)
        for scene in rows:
            for event in scene.get("events", []):
                identities.update(_event_identities(event))
    return identities, counts


def _source_rows(source_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    # Official eval is deliberately never traversed here.
    manifests = sorted((source_root / "train").glob("**/source_bank.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"no train source-bank manifests below {source_root}")
    rows: list[dict[str, Any]] = []
    for path in manifests:
        rows.extend(_read_jsonl(path))
    return rows, manifests


def main() -> None:
    args = parse_args()
    if args.sources_per_class < 1 or args.scene_count < 1:
        raise ValueError("sources-per-class and scene-count must be positive")
    if not 1 <= args.minimum_sources_per_class <= args.sources_per_class:
        raise ValueError("minimum-sources-per-class must be in [1, sources-per-class]")

    audit_path = args.semantic_audit.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("format") != "qces_v5_semantic_sufficiency_audit_v1":
        raise ValueError("unexpected semantic audit format")
    weak_rows = [
        row for row in audit["per_class"]
        if row["action"] == "enrich_existing_gold_reserve"
    ]
    weak_labels = [str(row["label"]) for row in weak_rows]
    if len(weak_labels) != 16 or len(weak_labels) != len(set(weak_labels)):
        raise RuntimeError(f"expected the frozen 16-class enrichment set, got {len(weak_labels)}")

    v5_root = args.v5_root.resolve()
    ontology = [
        line.strip() for line in (v5_root / "ontology_188.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ontology) != 188 or not set(weak_labels).issubset(ontology):
        raise RuntimeError("V5 ontology contract changed")
    label_to_id = {label: index for index, label in enumerate(ontology)}

    blocked, existing_scene_counts = _existing_identities(v5_root)
    raw_rows, source_manifests = _source_rows(args.source_root.resolve())
    candidate_by_label: dict[str, list[tuple[dict[str, Any], Path, set[tuple[str, str]]]]] = defaultdict(list)
    rejection = Counter()
    seen_source_ids: set[str] = set()
    for row in raw_rows:
        label = str(row.get("label") or row.get("coverage_label") or "")
        if label not in weak_labels:
            continue
        if not bool(row.get("accepted")) or str(row.get("acceptance_tier", "")).lower() != "gold":
            rejection["not_accepted_gold"] += 1
            continue
        official_split = str(row.get("metadata_split") or row.get("hf_split") or "").lower()
        if official_split != "train":
            rejection["not_official_train"] += 1
            continue
        source_id = str(row.get("source_id") or row.get("item_id") or "")
        if not source_id or source_id in seen_source_ids:
            rejection["duplicate_source_id"] += 1
            continue
        audio_path = _resolve_audio_path(row)
        if not audio_path.is_file():
            rejection["missing_audio"] += 1
            continue
        identities = _row_identities(row, audio_path)
        if identities & blocked:
            rejection["existing_v5_identity"] += 1
            continue
        seen_source_ids.add(source_id)
        candidate_by_label[label].append((row, audio_path, identities))

    selected_rows: list[dict[str, Any]] = []
    selected_identities: set[tuple[str, str]] = set()
    availability: dict[str, Any] = {}
    for label in weak_labels:
        candidates = sorted(
            candidate_by_label[label], key=lambda item: _quality_key(item[0]), reverse=True
        )
        selected_for_label: list[dict[str, Any]] = []
        for row, audio_path, identities in candidates:
            if identities & selected_identities:
                continue
            normalized = dict(row)
            normalized["audio_path"] = audio_path.as_posix()
            normalized["source_path"] = audio_path.as_posix()
            normalized["official_split"] = "train"
            normalized["cleanliness_passed"] = True
            normalized["audibility_passed"] = True
            normalized["cleanliness_tier"] = "gold"
            selected_for_label.append(normalized)
            selected_identities.update(identities)
            if len(selected_for_label) == args.sources_per_class:
                break
        availability[label] = {
            "eligible_unused_gold_train": len(candidates),
            "selected": len(selected_for_label),
        }
        if len(selected_for_label) < args.minimum_sources_per_class:
            raise RuntimeError(
                f"{label}: only {len(selected_for_label)} hard-identity-disjoint sources; "
                f"need at least {args.minimum_sources_per_class}"
            )
        selected_rows.extend(selected_for_label)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        selected_bank = staging / "selected_gold_train_sources.jsonl"
        _atomic_write_text(selected_bank, _jsonl(selected_rows))
        sources: list[CleanSource] = load_source_bank(selected_bank, require_audio_file=True)
        scenes, components, stats = v5.render_split(
            split="enrich",
            scene_count=args.scene_count,
            sources=sources,
            labels=weak_labels,
            label_to_id=label_to_id,
            staging=staging,
            final_root=output_dir,
            seed=args.seed,
            layout_trials=args.layout_trials,
        )
        scene_manifest = "detector_scene_manifest_targeted_enrich.jsonl"
        component_manifest = "event_components_targeted_enrich.jsonl"
        _atomic_write_text(staging / scene_manifest, _jsonl(scenes))
        _atomic_write_text(staging / component_manifest, _jsonl(components))
        _atomic_write_text(staging / "weak_labels_16.txt", "".join(f"{label}\n" for label in weak_labels))
        _atomic_write_text(staging / "ontology_188.txt", "".join(f"{label}\n" for label in ontology))

        rendered_identities = {
            identity
            for scene in scenes
            for event in scene["events"]
            for identity in _event_identities(event)
        }
        overlap = rendered_identities & blocked
        used_source_ids = {event["source_id"] for scene in scenes for event in scene["events"]}
        selected_source_ids = {str(row["source_id"]) for row in selected_rows}
        label_counts = Counter(event["label"] for scene in scenes for event in scene["events"])
        gates = {
            "exactly_16_target_classes": len(label_counts) == 16,
            "all_selected_sources_official_train": all(
                str(row["official_split"]).lower() == "train" for row in selected_rows
            ),
            "no_existing_v5_or_locked_test_identity_overlap": not overlap,
            "no_official_eval_manifest_traversed": all("/eval/" not in path.as_posix() for path in source_manifests),
            "all_selected_audio_exists": all(Path(str(row["audio_path"])).is_file() for row in selected_rows),
            "component_reconstruction_within_pcm_tolerance": (
                stats["maximum_reconstruction_error"] <= 2.0 / 32768.0
            ),
            "at_least_90_percent_selected_sources_rendered": (
                len(used_source_ids) / len(selected_source_ids) >= 0.90
            ),
        }
        if not all(gates.values()):
            raise RuntimeError(f"targeted enrichment quality gate failed: {gates}")

        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "data_selection": {
                "class_selection": "class-level V5 dev semantic-sufficiency audit",
                "individual_source_selection": "Gold quality rank only within each selected class",
                "qa_answers_used": False,
                "per_source_model_predictions_used": False,
                "official_eval_used": False,
                "locked_test_used_for_selection": False,
                "semantic_audit": str(audit_path),
                "semantic_audit_sha256": sha256_file(audit_path),
                "source_manifest_count": len(source_manifests),
                "sources_per_class": args.sources_per_class,
                "minimum_sources_per_class": args.minimum_sources_per_class,
                "selected_sources": len(selected_rows),
                "selected_sources_rendered": len(used_source_ids),
                "availability": availability,
                "rejections": dict(rejection),
            },
            "fixed_evaluation": {
                "v5_root": str(v5_root),
                "existing_scene_counts": existing_scene_counts,
                "dev_and_locked_test_rewritten": False,
                "hard_identity_overlap": len(overlap),
            },
            "render": stats,
            "weak_labels": weak_labels,
            "quality_gates": {"passed": all(gates.values()), "checks": gates},
            "artifacts": {},
        }
        artifact_names = [
            scene_manifest,
            component_manifest,
            "selected_gold_train_sources.jsonl",
            "weak_labels_16.txt",
            "ontology_188.txt",
        ]
        receipt["artifacts"] = {
            name: {"sha256": sha256_file(staging / name), "bytes": (staging / name).stat().st_size}
            for name in artifact_names
        }
        _atomic_write_text(staging / "build_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
        staging = None
        print(json.dumps({
            "output_dir": str(output_dir),
            "weak_classes": len(weak_labels),
            "selected_sources": len(selected_rows),
            "rendered_scenes": len(scenes),
            "rendered_events": len(components),
            "quality_gates_passed": True,
        }, indent=2))
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
