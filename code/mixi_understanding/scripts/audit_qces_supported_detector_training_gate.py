#!/usr/bin/env python3
"""Create the mandatory quota/integrity gate for detector-v2 training.

The receipt is intentionally derived from files, not a command-line promise.
It binds the ontology and exact train/dev/test manifest hashes, independently
rechecks hard source identity overlap, and requires an upstream materialization
receipt proving all 200 classes reached at least 100 train and 20 eval videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.clean_detector_splits import audit_strict_identity_overlap
from mixi_understanding.scripts.train_qces_supported_beats_detector_v2 import (
    GATE_FORMAT,
    _read_jsonl,
    _sha256,
    load_labels,
)


DEFAULT_ROOT = PROJECT_ROOT / "outputs/qces_supported_clean_protocol_v2_current_audio"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_ROOT / "detector_manifest_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=DEFAULT_ROOT / "detector_manifest_dev.jsonl")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_ROOT / "detector_manifest_test.jsonl")
    parser.add_argument(
        "--ontology",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1/ontology_200_supported.txt",
    )
    parser.add_argument(
        "--split-receipt",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_clean_detector_protocol_v1_current/clean_split_receipt.json",
    )
    parser.add_argument("--reannotation-receipt", type=Path, default=DEFAULT_ROOT / "reannotation_receipt.json")
    parser.add_argument(
        "--quota-receipt",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1/selection_receipt.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "detector_training_gate_v2.json")
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"receipt must be a JSON object: {path}")
    return payload


def _extract_quota(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize supported selector/materializer receipts to one strict gate."""

    explicit = payload.get("quota")
    if isinstance(explicit, Mapping):
        selected = int(explicit.get("selected_classes", 0))
        meeting = int(explicit.get("classes_meeting_train_eval_targets", 0))
        train_target = int(explicit.get("train_target_videos_per_label", 0))
        eval_target = int(explicit.get("eval_target_videos_per_label", 0))
    else:
        downloaded = payload.get("downloaded_material") or {}
        selected = int(downloaded.get("selected_classes", 0))
        meeting = int(
            downloaded.get(
                "classes_meeting_audioset_100_20",
                downloaded.get("classes_meeting_combined_train_test_scene_targets", 0),
            )
        )
        plan = payload.get("materialization_plan") or {}
        train_target = int((plan.get("train") or {}).get("target_videos_per_label", 0))
        eval_target = int((plan.get("eval") or {}).get("target_videos_per_label", 0))
    passes = selected == 200 and meeting == 200 and train_target >= 100 and eval_target >= 20
    return {
        "passes": passes,
        "selected_classes": selected,
        "classes_meeting_train_eval_targets": meeting,
        "train_target_videos_per_label": train_target,
        "eval_target_videos_per_label": eval_target,
    }


def _manifest_summary(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    ontology = set(labels)
    scene_support: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    for row in rows:
        seen: set[str] = set()
        for event in row.get("events") or ():
            label = str(event.get("label") or "")
            if label in ontology:
                seen.add(label)
                occurrences[label] += 1
        scene_support.update(seen)
    return {
        "rows": len(rows),
        "positive_classes": sum(scene_support[label] > 0 for label in labels),
        "scene_support_min": min((scene_support[label] for label in labels), default=0),
        "scene_support_median": sorted(scene_support[label] for label in labels)[len(labels) // 2],
        "occurrences": sum(occurrences.values()),
        "per_class_scene_support": {label: scene_support[label] for label in labels},
    }


def build_gate(args: argparse.Namespace) -> dict[str, Any]:
    ontology = args.ontology.resolve()
    labels = load_labels(ontology)
    manifest_paths = {
        "train": args.train_manifest.resolve(),
        "dev": args.dev_manifest.resolve(),
        "test": args.test_manifest.resolve(),
    }
    rows = {split: list(_read_jsonl(path)) for split, path in manifest_paths.items()}
    integrity_audit = audit_strict_identity_overlap(rows)
    summaries = {split: _manifest_summary(split_rows, labels) for split, split_rows in rows.items()}

    failures: list[str] = []
    split_receipt = _load_required_json(args.split_receipt.resolve())
    if split_receipt.get("passes") is not True:
        failures.append("upstream clean split receipt does not pass")
    upstream_identity = split_receipt.get("identity_audit") or {}
    if upstream_identity.get("passes") is not True or int(upstream_identity.get("overlap_count", -1)) != 0:
        failures.append("upstream clean split receipt does not prove zero identity overlap")

    reannotation = _load_required_json(args.reannotation_receipt.resolve())
    if int(reannotation.get("ontology_labels", 0)) != 200:
        failures.append("reannotation receipt does not bind 200 labels")
    if int(reannotation.get("cross_split_hard_overlap", -1)) != 0:
        failures.append("reannotation receipt reports cross-split overlap")
    output_hashes = reannotation.get("output_sha256") or {}
    for split, path in manifest_paths.items():
        if output_hashes.get(split) != _sha256(path):
            failures.append(f"reannotation hash mismatch for {split}")

    quota_payload = _load_required_json(args.quota_receipt.resolve())
    quota = _extract_quota(quota_payload)
    if not quota["passes"]:
        failures.append(
            "quota receipt does not prove 200/200 classes at >=100 train and >=20 eval videos"
        )
    if not integrity_audit["passes"]:
        failures.append("independent hard-identity audit found cross-split leakage")
    if summaries["train"]["positive_classes"] != 200:
        failures.append(
            f"train manifest contains positives for only {summaries['train']['positive_classes']}/200 classes"
        )
    eval_positive = {
        label
        for label in labels
        if summaries["dev"]["per_class_scene_support"][label] > 0
        or summaries["test"]["per_class_scene_support"][label] > 0
    }
    if len(eval_positive) != 200:
        failures.append(f"dev+test manifests contain positives for only {len(eval_positive)}/200 classes")

    return {
        "format": GATE_FORMAT,
        "passes": not failures,
        "paper_eligible": not failures,
        "failures": failures,
        "ontology": {"path": str(ontology), "sha256": _sha256(ontology), "labels": len(labels)},
        "manifests": {
            split: {"path": str(path), "sha256": _sha256(path), **summaries[split]}
            for split, path in manifest_paths.items()
        },
        "quota": quota,
        "integrity": {
            "passes": bool(integrity_audit["passes"]),
            "cross_split_hard_overlap": int(integrity_audit["overlap_count"]),
            "details": integrity_audit,
        },
        "upstream_receipts": {
            "split": {"path": str(args.split_receipt.resolve()), "sha256": _sha256(args.split_receipt.resolve())},
            "reannotation": {"path": str(args.reannotation_receipt.resolve()), "sha256": _sha256(args.reannotation_receipt.resolve())},
            "quota": {"path": str(args.quota_receipt.resolve()), "sha256": _sha256(args.quota_receipt.resolve())},
        },
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    payload = build_gate(args)
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()

