#!/usr/bin/env python3
"""Audit the native BEATs-Strong head on the supported QCES ontology.

This diagnostic uses only the clean AudioSet-derived dev partition.  Event
intervals are oracle inputs solely for representation analysis; no result from
this script is a deployable temporal-QA score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from data_util.audioset_classes import as_strong_train_classes  # noqa: E402
from mixi_understanding.qces.supported_ontology import (  # noqa: E402
    load_strong_metadata,
    safe_label,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (  # noqa: E402
    SceneDataset,
    SceneItem,
    collate,
    load_model,
    read_jsonl,
    strong_logits_from_waveform,
)


FRAME_HOP_SECONDS = 0.04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dev-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_clean_detector_protocol_v1_current/detector_manifest_dev.jsonl",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_supported_ontology_200_v1/ontology_200_supported.txt",
    )
    parser.add_argument(
        "--strong-metadata-dir",
        type=Path,
        default=PRETRAINED_ROOT / "hf_dataset_gen/metadata",
    )
    parser.add_argument("--audio-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_native_beats_supported_ontology/current/report.json",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
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


def _overlap(left: Any, right: Any) -> bool:
    return min(float(left.offset_seconds), float(right.offset_seconds)) > max(
        float(left.onset_seconds), float(right.onset_seconds)
    )


def _macro(correct: Counter[str], total: Counter[str]) -> float:
    values = [correct[label] / total[label] for label in total if total[label] > 0]
    return sum(values) / len(values) if values else 0.0


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    selected_labels = [
        line.strip()
        for line in args.ontology.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(selected_labels) != 200 or len(set(selected_labels)) != 200:
        raise SystemExit("supported ontology must contain exactly 200 unique labels")
    native_labels = [safe_label(value) for value in as_strong_train_classes]
    native_index = {label: index for index, label in enumerate(native_labels)}
    missing = sorted(set(selected_labels) - set(native_index))
    if missing:
        raise SystemExit(f"selected labels missing from native 447 head: {missing}")
    selected_native_indices = torch.tensor(
        [native_index[label] for label in selected_labels], dtype=torch.long
    )
    selected_index = {label: index for index, label in enumerate(selected_labels)}

    _, strong_events = load_strong_metadata(args.strong_metadata_dir)
    raw_rows = []
    for row in read_jsonl(args.clean_dev_manifest):
        source = str(row.get("protocol_source") or row.get("source_route") or "")
        if "audioset" not in source.lower():
            continue
        video_id = str(row.get("video_id") or "")
        events = list(strong_events["train"].get(video_id, ()))
        selected_events = [event for event in events if event.label in selected_index]
        if not selected_events:
            continue
        raw_rows.append((row, events, selected_events))
        if args.max_scenes and len(raw_rows) >= args.max_scenes:
            break
    if not raw_rows:
        raise SystemExit("no AudioSet dev scenes with selected labels")

    scene_items = [
        SceneItem(
            scene_id=str(row["scene_id"]),
            split="dev",
            mixture_path=str(row["mixture_path"]),
            duration_seconds=float(row.get("duration_seconds") or 10.0),
            sample_rate=int(row.get("sample_rate") or 32_000),
            events=tuple(),
        )
        for row, _, _ in raw_rows
    ]
    loader = DataLoader(
        SceneDataset(scene_items, audio_root=args.audio_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device)
    model = load_model(447, "BEATs_strong_1", device)
    model.eval()
    selected_native_indices = selected_native_indices.to(device)

    total = Counter()
    correct_top1 = Counter()
    correct_top5 = Counter()
    correct_top10 = Counter()
    multilabel_top1 = Counter()
    tier_total: dict[str, Counter[str]] = defaultdict(Counter)
    tier_correct: dict[str, Counter[str]] = defaultdict(Counter)
    scene_offset = 0
    for waveforms, batch_rows in loader:
        logits = strong_logits_from_waveform(model, waveforms.to(device))
        # Native contract is [B, 250, 447].
        logits = logits.index_select(2, selected_native_indices).cpu()
        for batch_index, scene in enumerate(batch_rows):
            source_row, all_events, target_events = raw_rows[scene_offset + batch_index]
            valid_frames = min(
                250,
                max(1, int(math.ceil(min(float(scene.duration_seconds), 10.0) / FRAME_HOP_SECONDS - 1e-5))),
            )
            for event in target_events:
                label = event.label
                start = max(0, min(valid_frames - 1, int(math.floor(event.onset_seconds / FRAME_HOP_SECONDS))))
                end = max(
                    start + 1,
                    min(valid_frames, int(math.ceil(event.offset_seconds / FRAME_HOP_SECONDS))),
                )
                pooled = logits[batch_index, start:end].amax(dim=0)
                ranking = pooled.argsort(descending=True)
                gold = selected_index[label]
                total[label] += 1
                correct_top1[label] += int(int(ranking[0]) == gold)
                correct_top5[label] += int(bool((ranking[:5] == gold).any()))
                correct_top10[label] += int(bool((ranking[:10] == gold).any()))

                overlapping_selected = {
                    selected_index[other.label]
                    for other in all_events
                    if other.label in selected_index and _overlap(event, other)
                }
                multilabel_top1[label] += int(int(ranking[0]) in overlapping_selected)
                overlaps_any_different = any(
                    other.label != label and _overlap(event, other) for other in all_events
                )
                overlaps_selected_different = any(
                    other.label != label
                    and other.label in selected_index
                    and _overlap(event, other)
                    for other in all_events
                )
                tiers = ["all"]
                if not overlaps_selected_different:
                    tiers.append("no_selected_label_overlap")
                if not overlaps_any_different:
                    tiers.append("fully_isolated_annotation")
                for tier in tiers:
                    tier_total[tier][label] += 1
                    tier_correct[tier][label] += int(int(ranking[0]) == gold)
        scene_offset += len(batch_rows)
        print(f"scenes={scene_offset}/{len(raw_rows)} events={sum(total.values())}", flush=True)

    def metrics_for(correct: Counter[str], denominator: Counter[str]) -> dict[str, Any]:
        count = sum(denominator.values())
        return {
            "events": count,
            "classes": sum(value > 0 for value in denominator.values()),
            "micro_accuracy": sum(correct.values()) / count if count else 0.0,
            "macro_accuracy_observed_classes": _macro(correct, denominator),
        }

    report = {
        "format": "qces_native_beats_supported_ontology_audit_v1",
        "status": "oracle_window_representation_diagnostic_not_deployable",
        "data_contract": "clean source-disjoint AudioSet-train dev partition; official timestamps; native frozen 447-class BEATs-Strong head",
        "scenes": len(raw_rows),
        "selected_labels": len(selected_labels),
        "native_labels": len(native_labels),
        "exact_target": {
            "top1": metrics_for(correct_top1, total),
            "top5": metrics_for(correct_top5, total),
            "top10": metrics_for(correct_top10, total),
        },
        "overlap_aware_top1": metrics_for(multilabel_top1, total),
        "tiers_exact_top1": {
            tier: metrics_for(tier_correct[tier], tier_total[tier])
            for tier in sorted(tier_total)
        },
        "inputs": {
            "manifest": str(args.clean_dev_manifest.resolve()),
            "manifest_sha256": _sha256(args.clean_dev_manifest.resolve()),
            "ontology": str(args.ontology.resolve()),
            "ontology_sha256": _sha256(args.ontology.resolve()),
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), **report["exact_target"], "tiers": report["tiers_exact_top1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
