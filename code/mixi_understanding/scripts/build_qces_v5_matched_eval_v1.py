#!/usr/bin/env python3
"""Render a V5-profile evaluation view from official AudioSet eval sources.

The historical locked test is retained unchanged as a short-event stress view.
This sidecar uses the already-frozen V5 renderer/profile/seed policy to create
an IID-render evaluation view.  It never changes train/dev and never selects a
source using model predictions.  Official-eval sources may overlap the legacy
test by design, but can never overlap train/dev/enrichment identities.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.clean_evidence_scenes import (  # noqa: E402
    CleanSource,
    _atomic_write_text,
    load_source_bank,
    partition_sources,
    sha256_file,
)
from mixi_understanding.scripts import build_qces_overlap_gold_natural_v3 as v3  # noqa: E402
from mixi_understanding.scripts import build_qces_tiered_realistic_v5 as v5  # noqa: E402


FORMAT = "qces_v5_matched_eval_v1"


def parse_args() -> argparse.Namespace:
    original = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    v5_root = PROJECT_ROOT / "outputs/qces_full188_tiered_realistic_v5"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=original / "source_bank_accepted.jsonl")
    parser.add_argument("--source-ontology", type=Path, default=original / "ontology_191.txt")
    parser.add_argument("--v5-root", type=Path, default=v5_root)
    parser.add_argument("--enrichment-root", type=Path, default=Path("/var/tmp/qces_v5_targeted_enrichment_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("/var/tmp/qces_v5_matched_eval_v1"))
    parser.add_argument("--scene-count", type=int, default=573)
    parser.add_argument("--seed", type=int, default=v5.DEFAULT_SEED)
    parser.add_argument("--layout-trials", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def identities(sources: Sequence[CleanSource]) -> set[tuple[str, str]]:
    return {identity for source in sources for identity in source.hard_identities}


def manifest_identities(path: Path) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for line in path.resolve().read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for event in json.loads(line).get("events", []):
            for name, value in (
                ("source_id", event.get("source_id")),
                ("video_id", event.get("source_video_id") or event.get("video_id")),
                ("source_sha256", event.get("source_sha256")),
                ("audio_path", event.get("source_path") or event.get("audio_path")),
            ):
                if value:
                    result.add((name, str(value)))
    return result


def main() -> None:
    args = parse_args()
    v5_root = args.v5_root.resolve()
    frozen_receipt_path = v5_root / "build_receipt.json"
    frozen_receipt = json.loads(frozen_receipt_path.read_text(encoding="utf-8"))
    if frozen_receipt.get("format") != v5.FORMAT:
        raise ValueError("frozen V5 build contract unavailable")
    frozen_labels = [
        line.strip() for line in (v5_root / "ontology_188.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_labels = [
        line.strip() for line in args.source_ontology.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources = load_source_bank(args.source_bank.resolve(), require_audio_file=True)
    partitioned, partition_receipt = partition_sources(
        sources, source_labels, seed=2041, dev_fraction=0.20
    )
    test_sources = [
        source for source in partitioned["test"]
        if source.label in set(frozen_labels) and v3._gold(source)
    ]
    available = {source.label for source in test_sources}
    if missing := sorted(set(frozen_labels) - available):
        raise RuntimeError(f"official eval lacks frozen labels: {missing}")

    train_dev_ids = manifest_identities(v5_root / "detector_scene_manifest_tiered_train.jsonl")
    train_dev_ids.update(manifest_identities(v5_root / "detector_scene_manifest_tiered_dev.jsonl"))
    enrichment_manifest = args.enrichment_root.resolve() / "detector_scene_manifest_targeted_enrich.jsonl"
    train_dev_ids.update(manifest_identities(enrichment_manifest))
    pre_render_overlap = identities(test_sources) & train_dev_ids
    if pre_render_overlap:
        raise RuntimeError(f"official-eval source overlaps training identities: {len(pre_render_overlap)}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        label_to_id = {label: index for index, label in enumerate(frozen_labels)}
        scenes, components, stats = v5.render_split(
            split="matched_eval",
            scene_count=args.scene_count,
            sources=test_sources,
            labels=frozen_labels,
            label_to_id=label_to_id,
            staging=staging,
            final_root=output_dir,
            seed=args.seed,
            layout_trials=args.layout_trials,
        )
        manifest_name = "detector_scene_manifest_matched_eval.jsonl"
        component_name = "event_components_matched_eval.jsonl"
        _atomic_write_text(staging / manifest_name, _jsonl(scenes))
        _atomic_write_text(staging / component_name, _jsonl(components))
        _atomic_write_text(staging / "ontology_188.txt", "".join(f"{label}\n" for label in frozen_labels))

        rendered_ids = manifest_identities(staging / manifest_name)
        post_render_overlap = rendered_ids & train_dev_ids
        overlap_bins = stats["event_overlap_fraction"]["bins"]
        light = overlap_bins["zero"]["fraction"] + overlap_bins["light_(0,.25]"]["fraction"]
        checks = {
            "all_188_classes_present": stats["class_coverage"]["classes"] == 188,
            "official_eval_only": all(source.official_split == "eval" for source in test_sources),
            "no_train_dev_enrichment_identity_overlap": not post_render_overlap,
            "maximum_concurrency_at_most_3": stats["maximum_concurrency"]["maximum"] <= 3,
            "heavy_overlap_fraction_at_most_0_30": overlap_bins["heavy_(.75,1]"]["fraction"] <= 0.30,
            "zero_or_light_overlap_fraction_at_least_0_40": light >= 0.40,
            "sir_below_minus_10_fraction_at_most_0_10": stats["active_sir_db"]["bins"]["below_-10"]["fraction"] <= 0.10,
            "no_tied_onsets": stats["scenes_with_tied_onsets"] == 0,
            "component_reconstruction_within_pcm_tolerance": stats["maximum_reconstruction_error"] <= 2.0 / 32768.0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"matched eval quality gate failed: {checks}")

        legacy_manifest = v5_root / "detector_scene_manifest_locked_test_filtered.jsonl"
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "role": "IID-render evaluation paired with unchanged legacy short-event stress test",
            "construction_timing_note": "built after stress diagnosis; renderer/profile copied exactly from frozen V5 and no model prediction used",
            "policy": {
                "renderer": v5.FORMAT,
                "profiles": v5.PROFILES,
                "model_predictions_used": False,
                "qa_answers_used": False,
                "official_eval_sources_only": True,
                "train_dev_or_enrichment_rewritten": False,
                "legacy_locked_test_rewritten": False,
                "legacy_and_matched_views_may_share_eval_sources": True,
            },
            "data": stats,
            "source_pool": {
                "official_eval_gold_sources": len(test_sources),
                "classes": len(available),
                "partition_policy": partition_receipt["policy"],
            },
            "identity_audit": {
                "matched_eval_vs_train_dev_enrichment": len(post_render_overlap),
                "legacy_eval_vs_matched_eval": len(manifest_identities(legacy_manifest) & rendered_ids),
            },
            "quality_gates": {"passed": all(checks.values()), "checks": checks},
            "artifacts": {},
        }
        names = [manifest_name, component_name, "ontology_188.txt"]
        receipt["artifacts"] = {
            name: {"sha256": sha256_file(staging / name), "bytes": (staging / name).stat().st_size}
            for name in names
        }
        receipt["frozen_v5_receipt_sha256"] = sha256_file(frozen_receipt_path)
        receipt["legacy_test_manifest_sha256"] = sha256_file(legacy_manifest)
        _atomic_write_text(
            staging / "build_receipt.json",
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
        staging = None
        print(json.dumps({
            "output_dir": str(output_dir), "scenes": len(scenes),
            "events": len(components), "classes": stats["class_coverage"]["classes"],
            "train_overlap": len(post_render_overlap), "quality_gates_passed": True,
        }, indent=2))
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
