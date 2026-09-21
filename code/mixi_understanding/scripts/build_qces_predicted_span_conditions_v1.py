#!/usr/bin/env python3
"""Materialize real temporal-v2 proposal conditions for separator adaptation.

The frozen temporal model is run independently on source-disjoint train and
development scenes.  Each gold event is matched to the retained proposal with
maximum temporal IoU.  Gold labels never enter proposal generation; they are
used only afterwards to form ordinary supervised proposal/event matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
)
from mixi_understanding.scripts.evaluate_qces_polyphonic_deployable_decoder_v1 import (
    FIXED_AUDIO_SECONDS,
    collect,
    interval_iou,
)
from mixi_understanding.scripts.train_qces_polyphonic_query_branch_v1 import (
    POLYPHONIC_CHECKPOINT_FORMAT,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _device,
    load_scene_list,
)


FORMAT = "qces_predicted_span_conditions_v1"
RATE = 16_000
SCENE_SAMPLES = int(round(FIXED_AUDIO_SECONDS * RATE))


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    components = PROJECT_ROOT / "outputs/qces_full191_overlap_components_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=base / "polyphonic_query_branch_v2/polyphonic_query_branch_best.pt",
    )
    parser.add_argument(
        "--decoder-receipt", type=Path,
        default=base / "polyphonic_deployable_decoder_v1/receipt.json",
    )
    parser.add_argument(
        "--train-index", type=Path, default=base / "dense_overlap_query_train_v2/index.json",
    )
    parser.add_argument(
        "--dev-index", type=Path, default=base / "dense_overlap_query_dev_v2/index.json",
    )
    parser.add_argument(
        "--train-scenes", type=Path, default=data / "scene_ids_overlap_train.txt",
    )
    parser.add_argument(
        "--dev-scenes", type=Path, default=data / "scene_ids_overlap_dev.txt",
    )
    parser.add_argument(
        "--train-components", type=Path, default=components / "event_components_train.jsonl",
    )
    parser.add_argument(
        "--dev-components", type=Path, default=components / "event_components_dev.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_predicted_span_conditions_v1",
    )
    parser.add_argument("--minimum-match-iou", type=float, default=0.30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _match_split(
    *, store: DenseFeatureStore, scene_ids: Sequence[str], component_path: Path,
    predictions: Mapping[str, Mapping[str, Any]], threshold: float, minimum_iou: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = set(scene_ids)
    rows = [row for row in _read_jsonl(component_path) if str(row["scene_id"]) in allowed]
    output: list[dict[str, Any]] = []
    retained_ious: list[float] = []
    unrestricted_ious: list[float] = []
    available = 0
    for row in rows:
        scene_id = str(row["scene_id"])
        gold = (
            float(row["onset_sample"]) / SCENE_SAMPLES,
            float(row["offset_sample"]) / SCENE_SAMPLES,
        )
        slots = [dict(slot) for slot in predictions[scene_id]["slots"]]
        best_any = max(slots, key=lambda slot: interval_iou((slot["start"], slot["end"]), gold))
        best_any_iou = interval_iou((best_any["start"], best_any["end"]), gold)
        unrestricted_ious.append(best_any_iou)
        retained = [slot for slot in slots if float(slot["score"]) >= threshold]
        best = None
        best_iou = 0.0
        if retained:
            best = max(
                retained,
                key=lambda slot: (
                    interval_iou((slot["start"], slot["end"]), gold), float(slot["score"])
                ),
            )
            best_iou = interval_iou((best["start"], best["end"]), gold)
        condition_available = best is not None and best_iou >= minimum_iou
        condition_onset = None
        condition_offset = None
        if condition_available:
            condition_onset = max(0, min(SCENE_SAMPLES - 1, int(round(float(best["start"]) * SCENE_SAMPLES))))
            condition_offset = max(
                condition_onset + 1,
                min(SCENE_SAMPLES, int(round(float(best["end"]) * SCENE_SAMPLES))),
            )
            available += 1
        retained_ious.append(best_iou)
        augmented = dict(row)
        augmented.update({
            "condition_available": condition_available,
            "condition_onset_sample": condition_onset,
            "condition_offset_sample": condition_offset,
            "condition_iou": float(best_iou),
            "condition_score": None if best is None else float(best["score"]),
            "condition_slot_index": None if best is None else int(best["slot_index"]),
            "retained_proposals": len(retained),
            "best_unrestricted_iou": float(best_any_iou),
        })
        output.append(augmented)
    array = np.asarray(retained_ious, dtype=np.float64)
    unrestricted = np.asarray(unrestricted_ious, dtype=np.float64)
    summary = {
        "events": len(rows),
        "condition_available": available,
        "condition_available_rate": available / max(len(rows), 1),
        "retained_best_iou30_recall": float(np.mean(array >= 0.30)),
        "retained_best_iou50_recall": float(np.mean(array >= 0.50)),
        "retained_best_iou70_recall": float(np.mean(array >= 0.70)),
        "unrestricted_best_iou30_recall": float(np.mean(unrestricted >= 0.30)),
        "unrestricted_best_iou50_recall": float(np.mean(unrestricted >= 0.50)),
        "median_retained_best_iou": float(np.median(array)),
    }
    return output, summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != POLYPHONIC_CHECKPOINT_FORMAT:
        raise ValueError("invalid temporal checkpoint format")
    store = DenseFeatureStore([args.train_index.resolve(), args.dev_index.resolve()], cache_size=32)
    labels = list(store.labels or [])
    if list(checkpoint["labels"]) != labels:
        raise ValueError("checkpoint/dense label order mismatch")
    model = RelationalEventSlotsV1(RelationalEventSlotsV1Config(**checkpoint["config"])).to(
        _device(args.device)
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    decoder = json.loads(args.decoder_receipt.resolve().read_text(encoding="utf-8"))
    threshold = float(decoder["dev_split"]["locked_threshold"])
    split_specs = {
        "train": (load_scene_list(args.train_scenes.resolve()), args.train_components.resolve()),
        "dev": (load_scene_list(args.dev_scenes.resolve()), args.dev_components.resolve()),
    }
    summaries: dict[str, Any] = {}
    manifests: dict[str, str] = {}
    for split, (scene_ids, components) in split_specs.items():
        print(json.dumps({"split": split, "stage": "collect", "scenes": len(scene_ids)}), flush=True)
        predictions = collect(model, store, scene_ids, next(model.parameters()).device, args.batch_size)
        rows, summary = _match_split(
            store=store, scene_ids=scene_ids, component_path=components,
            predictions=predictions, threshold=threshold, minimum_iou=args.minimum_match_iou,
        )
        manifest = output_dir / f"event_components_{split}.jsonl"
        _atomic_jsonl(manifest, rows)
        summaries[split] = summary
        manifests[split] = str(manifest)
        print(json.dumps({"split": split, **summary}), flush=True)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "decoder_receipt": str(args.decoder_receipt.resolve()),
        "locked_objectness_threshold": threshold,
        "minimum_match_iou": args.minimum_match_iou,
        "proposal_generation": "frozen temporal-v2 K=8; label-free generation; retained by locked threshold",
        "supervised_matching": "maximum IoU between retained proposals and each gold event",
        "manifests": manifests,
        "splits": summaries,
    }
    _atomic_json(output_dir / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
