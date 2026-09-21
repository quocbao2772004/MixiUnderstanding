#!/usr/bin/env python3
"""Audit V5 realistic-main against the frozen V4 stress distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


FORMAT = "qces_v5_main_stress_distribution_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--stress-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    scenes = list(rows)
    events = [event for scene in scenes for event in scene.get("events") or []]
    onset_quarters = [0, 0, 0, 0]
    for event in events:
        onset = float(event["onset_seconds"])
        onset_quarters[min(3, int(onset // 2.5))] += 1
    label_counts = Counter(str(event["label"]) for event in events)
    sources = {str(event["source_id"]) for event in events}
    return {
        "scenes": len(scenes),
        "events": len(events),
        "classes": len(label_counts),
        "minimum_events_per_class": min(label_counts.values()),
        "maximum_events_per_class": max(label_counts.values()),
        "unique_sources": len(sources),
        "source_ids": sources,
        "onset_quarter_counts": onset_quarters,
        "onset_quarter_fractions": [value / max(len(events), 1) for value in onset_quarters],
        "maximum_onset_seconds": max(float(event["onset_seconds"]) for event in events),
        "maximum_offset_seconds": max(float(event["offset_seconds"]) for event in events),
        "tied_onset_scenes": sum(
            len(scene.get("events") or [])
            != len({float(event["onset_seconds"]) for event in scene.get("events") or []})
            for scene in scenes
        ),
        "profile_scenes": dict(Counter(str(scene.get("layout_profile") or "unspecified") for scene in scenes)),
    }


def _public(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "source_ids"}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    main_root = args.main_root.resolve()
    stress_root = args.stress_root.resolve()
    build_receipt_path = main_root / "build_receipt.json"
    build_receipt = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    split_results: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    main_summaries: dict[str, dict[str, Any]] = {}
    stress_summaries: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        main_manifest = main_root / f"detector_scene_manifest_tiered_{split}.jsonl"
        stress_manifest = stress_root / f"detector_scene_manifest_overlap_{split}.jsonl"
        main_summary = _summarize(_read_jsonl(main_manifest))
        stress_summary = _summarize(_read_jsonl(stress_manifest))
        main_summaries[split] = main_summary
        stress_summaries[split] = stress_summary
        fractions = main_summary["onset_quarter_fractions"]
        split_gates = {
            "all_188_classes_present": main_summary["classes"] == 188,
            "no_tied_onsets": main_summary["tied_onset_scenes"] == 0,
            "first_quarter_not_dominant": fractions[0] <= 0.55,
            "second_half_has_at_least_25_percent": fractions[2] + fractions[3] >= 0.25,
            "last_quarter_has_at_least_5_percent": fractions[3] >= 0.05,
            "events_reach_last_second": main_summary["maximum_offset_seconds"] >= 9.0,
            "main_and_stress_scene_counts_match": main_summary["scenes"] == stress_summary["scenes"],
            "builder_quality_gate_passed": bool(build_receipt["quality_gates"][split]["passed"]),
        }
        for name, passed in split_gates.items():
            gates[f"{split}:{name}"] = passed
        split_results[split] = {
            "main": _public(main_summary),
            "stress": _public(stress_summary),
            "gates": split_gates,
        }
    source_overlap = len(main_summaries["train"]["source_ids"] & main_summaries["dev"]["source_ids"])
    gates["main_train_dev_source_overlap_zero"] = source_overlap == 0
    payload = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "main_root": str(main_root),
        "stress_root": str(stress_root),
        "build_receipt_sha256": _sha256(build_receipt_path),
        "source_overlap_train_dev": source_overlap,
        "splits": split_results,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "interpretation": "V5 is the realistic full-timeline main tier; V4 is retained unchanged as the dense early-timeline stress tier.",
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    if not payload["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
