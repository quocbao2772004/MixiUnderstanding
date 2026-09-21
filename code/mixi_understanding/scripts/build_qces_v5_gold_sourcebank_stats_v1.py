#!/usr/bin/env python3
"""Build leakage-safe extra Gold clean statistics for V5 group experts.

Only accepted Gold rows from the primary *train* source bank are eligible.
Any source/video already used by V5 train, V5 dev, or matched official-eval is
removed.  The resulting cache therefore adds source diversity instead of
duplicating the synthetic training components or leaking evaluation identity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models.prediction_wrapper as prediction_wrapper_module
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper
from mixi_understanding.qces.clean_evidence_scenes import _atomic_write_text
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    SceneDataset,
    SceneItem,
    collate,
    load_ontology,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _atomic_torch,
    _sha256_file,
)


FORMAT = "qces_v5_gold_sourcebank_stats_v1"
NUM_FRAMES = 250
FIXED_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
    data = Path("/var/tmp/qces_full188_tiered_realistic_v5")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-index", type=Path, default=PROJECT_ROOT / "data_full/index/accepted_samples.jsonl")
    parser.add_argument("--ontology", type=Path, default=PROJECT_ROOT / "outputs/qces_full188_tiered_realistic_v5/ontology_188.txt")
    parser.add_argument("--v5-train-manifest", type=Path, default=data / "detector_scene_manifest_tiered_train.jsonl")
    parser.add_argument("--v5-dev-manifest", type=Path, default=data / "detector_scene_manifest_tiered_dev.jsonl")
    parser.add_argument("--matched-manifest", type=Path, default=Path("/var/tmp/qces_v5_matched_eval_v1/detector_scene_manifest_matched_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/var/tmp/qces_v5_gold_sourcebank_stats_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def blocked_identities(paths: Sequence[Path]) -> tuple[set[str], set[str], dict[str, Any]]:
    source_ids: set[str] = set()
    video_ids: set[str] = set()
    audit: dict[str, Any] = {}
    for path in paths:
        scenes = events = 0
        for scene in read_jsonl(path):
            scenes += 1
            for event in scene.get("events", []):
                events += 1
                if event.get("source_id"):
                    source_ids.add(str(event["source_id"]))
                value = event.get("source_video_id") or event.get("video_id")
                if value:
                    video_ids.add(str(value))
        audit[path.name] = {"scenes": scenes, "events": events}
    return source_ids, video_ids, audit


def resolve_audio(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def selected_rows(
    index_path: Path,
    labels: Sequence[str],
    blocked_sources: set[str],
    blocked_videos: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ontology = set(labels)
    rows: list[dict[str, Any]] = []
    rejection: Counter[str] = Counter()
    seen_sources: set[str] = set()
    for row in read_jsonl(index_path):
        label = str(row.get("label") or "")
        if label not in ontology:
            rejection["outside_188_ontology"] += 1
            continue
        if str(row.get("split") or "").lower() != "train":
            rejection["not_primary_train"] += 1
            continue
        if str(row.get("quality_tier") or "").lower() != "gold":
            rejection["not_gold"] += 1
            continue
        source_id = str(row.get("sample_id") or "")
        video_id = str(row.get("video_id") or "")
        if not source_id or source_id in seen_sources:
            rejection["missing_or_duplicate_source"] += 1
            continue
        if source_id in blocked_sources:
            rejection["already_used_source_id"] += 1
            continue
        if video_id and video_id in blocked_videos:
            rejection["already_used_video_id"] += 1
            continue
        path = resolve_audio(str(row.get("audio_path") or ""))
        if not path.is_file():
            rejection["missing_audio"] += 1
            continue
        normalized = dict(row)
        normalized["audio_path"] = path.as_posix()
        normalized["label_id"] = labels.index(label)
        rows.append(normalized)
        seen_sources.add(source_id)
    rows.sort(key=lambda row: (int(row["label_id"]), str(row["sample_id"])))
    return rows, dict(rejection)


def scene_rows(rows: Sequence[Mapping[str, Any]]) -> list[SceneItem]:
    result: list[SceneItem] = []
    for row in rows:
        onset = max(0.0, min(FIXED_SECONDS, float(row.get("active_onset_seconds") or 0.0)))
        offset = max(onset + 0.04, min(FIXED_SECONDS, float(row.get("active_offset_seconds") or row.get("duration_seconds") or FIXED_SECONDS)))
        event = {
            "event_kind": "semantic",
            "label": str(row["label"]),
            "label_id": int(row["label_id"]),
            "onset_seconds": onset,
            "offset_seconds": offset,
        }
        result.append(
            SceneItem(
                scene_id=str(row["sample_id"]),
                split="extra_gold_train",
                mixture_path=str(row["audio_path"]),
                duration_seconds=FIXED_SECONDS,
                sample_rate=16_000,
                events=(event,),
            )
        )
    return result


@torch.inference_mode()
def export_stats(
    backbone: PredictionsWrapper,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> dict[str, Any]:
    scenes = scene_rows(rows)
    data_loader = DataLoader(
        SceneDataset(scenes, audio_root=Path("/"), fixed_seconds=FIXED_SECONDS),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )
    output: list[torch.Tensor] = []
    processed = 0
    for waveforms, batch_rows in data_loader:
        audio = waveforms.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            mel = backbone.mel_forward(audio)
            features = backbone.model(mel)
        if int(features.shape[1]) != NUM_FRAMES:
            features = F.interpolate(
                features.transpose(1, 2), size=NUM_FRAMES, mode="linear", align_corners=False
            ).transpose(1, 2)
        for index, scene in enumerate(batch_rows):
            event = scene.events[0]
            start = max(0, min(NUM_FRAMES - 1, int(float(event["onset_seconds"]) / FIXED_SECONDS * NUM_FRAMES)))
            end = max(start + 1, min(NUM_FRAMES, int(math.ceil(float(event["offset_seconds"]) / FIXED_SECONDS * NUM_FRAMES))))
            span = features[index, start:end].float()
            output.append(
                torch.cat((span.mean(0), span.amax(0), span.std(0, unbiased=False))).cpu().half()
            )
        processed += len(batch_rows)
        if processed == len(batch_rows) or processed == len(rows) or processed % 256 < len(batch_rows):
            print(f"gold_sourcebank_atst_stats {processed}/{len(rows)}", flush=True)
    return {
        "format": FORMAT + "_cache",
        "features": torch.stack(output),
        "targets": torch.tensor([int(row["label_id"]) for row in rows], dtype=torch.long),
        "source_id": [str(row["sample_id"]) for row in rows],
        "video_id": [str(row.get("video_id") or "") for row in rows],
        "label": [str(row["label"]) for row in rows],
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_ontology(args.ontology.resolve())
    if len(labels) != 188:
        raise ValueError("expected frozen 188-class ontology")
    manifest_paths = [
        args.v5_train_manifest.resolve(),
        args.v5_dev_manifest.resolve(),
        args.matched_manifest.resolve(),
    ]
    blocked_sources, blocked_videos, blocked_audit = blocked_identities(manifest_paths)
    rows, rejection = selected_rows(
        args.accepted_index.resolve(), labels, blocked_sources, blocked_videos
    )
    if len(rows) < 4000:
        raise RuntimeError(f"unexpectedly small unused Gold source tier: {len(rows)}")
    selected_manifest = output_dir / "selected_unused_gold_train.jsonl"
    _atomic_write_text(
        selected_manifest,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    counts = Counter(str(row["label"]) for row in rows)
    missing = [label for label in labels if counts[label] == 0]

    device = make_device(args.device)
    prediction_wrapper_module.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    backbone = PredictionsWrapper(
        ATSTWrapper(),
        checkpoint="ATST-F_strong_1",
        n_classes_strong=len(labels),
        n_classes_weak=len(labels),
        seq_model_type=None,
        head_type="linear",
    ).to(device)
    backbone.eval().requires_grad_(False)
    cache = export_stats(
        backbone,
        rows,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )
    cache_path = output_dir / "gold_stats_train.pt"
    _atomic_torch(cache, cache_path)
    overlap_sources = set(cache["source_id"]) & blocked_sources
    overlap_videos = (set(cache["video_id"]) - {""}) & blocked_videos
    gates = {
        "at_least_4000_extra_gold_sources": len(rows) >= 4000,
        "at_least_180_observed_classes": len(counts) >= 180,
        "zero_source_id_overlap": not overlap_sources,
        "zero_video_id_overlap": not overlap_videos,
        "feature_target_count_match": len(cache["features"]) == len(cache["targets"]) == len(rows),
        "finite_features": bool(torch.isfinite(cache["features"]).all()),
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "accepted Gold primary train only; exclude every source/video used by V5 train/dev/matched",
        "qa_answer_used": False,
        "model_accuracy_used_for_individual_selection": False,
        "data": {
            "selected_sources": len(rows),
            "observed_classes": len(counts),
            "missing_classes": missing,
            "minimum_nonzero_per_class": min(counts.values()),
            "median_per_class": float(torch.tensor(list(counts.values()), dtype=torch.float32).median()),
            "maximum_per_class": max(counts.values()),
            "per_class": dict(sorted(counts.items())),
            "rejections": rejection,
            "blocked_manifest_audit": blocked_audit,
        },
        "gates": {"passed": all(gates.values()), "checks": gates},
        "artifacts": {
            "accepted_index_sha256": _sha256_file(args.accepted_index.resolve()),
            "selected_manifest": str(selected_manifest),
            "selected_manifest_sha256": _sha256_file(selected_manifest),
            "cache": str(cache_path),
            "cache_sha256": _sha256_file(cache_path),
            "blocked_manifests": {
                path.name: _sha256_file(path) for path in manifest_paths
            },
        },
    }
    receipt_path = output_dir / "receipt.json"
    _atomic_json(receipt, receipt_path)
    print(json.dumps({
        "complete": True,
        "selected_sources": len(rows),
        "observed_classes": len(counts),
        "missing_classes": missing,
        "gates": gates,
        "cache": str(cache_path),
        "receipt": str(receipt_path),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
