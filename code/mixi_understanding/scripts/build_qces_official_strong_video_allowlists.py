#!/usr/bin/env python3
"""Export the complete official AudioSet-Strong video-ID candidate universe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from mixi_understanding.qces.audioset_availability_index import atomic_bytes, atomic_json
from mixi_understanding.qces.supported_ontology import load_strong_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_DIR = (
    PROJECT_ROOT / "code/baseline/PretrainedSED/hf_dataset_gen/metadata"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_official_strong_video_allowlists_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_dir = args.metadata_dir.resolve()
    output_dir = args.output_dir.resolve()
    inputs = [
        metadata_dir / "class_labels_indices_strong.csv",
        metadata_dir / "audioset_train_strong.csv",
        metadata_dir / "audioset_eval_strong.csv",
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"missing official strong metadata: {missing}")
    _, events_by_split = load_strong_metadata(metadata_dir)
    ids = {
        split: sorted(str(video_id) for video_id in events_by_split[split])
        for split in ("train", "eval")
    }
    overlap = set(ids["train"]) & set(ids["eval"])
    if overlap:
        raise SystemExit(
            f"official strong train/eval video overlap: {len(overlap)}; "
            f"examples={sorted(overlap)[:10]}"
        )
    multi_segment: dict[str, list[dict[str, object]]] = {"train": [], "eval": []}
    for split in ("train", "eval"):
        for video_id, events in events_by_split[split].items():
            segment_ids = sorted({str(event.segment_id) for event in events})
            if len(segment_ids) != 1:
                multi_segment[split].append(
                    {"video_id": video_id, "segment_ids": segment_ids}
                )
    if any(multi_segment.values()):
        raise SystemExit(
            "video_id does not uniquely identify one official strong segment: "
            f"train={len(multi_segment['train'])}, eval={len(multi_segment['eval'])}"
        )
    output_paths = {
        "train": output_dir / "official_strong_video_ids_train.txt",
        "eval": output_dir / "official_strong_video_ids_eval.txt",
    }
    receipt_path = output_dir / "official_strong_allowlist_receipt.json"
    existing = [
        str(path) for path in [*output_paths.values(), receipt_path] if path.exists()
    ]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs exist; pass --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, path in output_paths.items():
        atomic_bytes(path, ("\n".join(ids[split]) + "\n").encode("utf-8"))
    receipt = {
        "format": "qces_official_strong_video_allowlists_v1",
        "audit_passes": True,
        "input_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in inputs
        ],
        "video_ids": {split: len(ids[split]) for split in ("train", "eval")},
        "strong_events": {
            split: sum(len(events) for events in events_by_split[split].values())
            for split in ("train", "eval")
        },
        "train_eval_video_overlap": 0,
        "video_ids_with_multiple_official_segments": {
            "train": 0,
            "eval": 0,
        },
        "outputs": {
            split: {
                "path": str(path),
                "sha256": _sha256(path),
                "rows": len(ids[split]),
            }
            for split, path in output_paths.items()
        },
        "identity_contract": (
            "within each official split, one video_id maps to exactly one "
            "AudioSet segment_id; mirror label identity is audited separately"
        ),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

