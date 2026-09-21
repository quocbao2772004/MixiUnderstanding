#!/usr/bin/env python3
"""Export sharded, fixed-grid dense features from a trained QCES detector.

This is a v2 sidecar exporter.  It deliberately leaves the v1 frame-
probability exporter unchanged.  Each scene is represented on the exact
250-frame (40 ms) grid used by PretrainedSED and stores the representation
immediately before the detector's strong linear head.

The dataset is committed transactionally: shards use a new run id and
``index.json`` is atomically replaced only after every shard has been written
and validated.  An interrupted overwrite therefore cannot invalidate the
previous index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from torch.utils.data import DataLoader

from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    SceneDataset,
    SceneItem,
    collate,
    interpolate_sequence,
    load_model,
    load_ontology,
    make_device,
    read_jsonl,
)


FORMAT = "qces_detector_dense_features_v2"
SHARD_FORMAT = "qces_detector_dense_features_shard_v2"
FIXED_AUDIO_SECONDS = 10.0
NUM_FRAMES = 250
FRAME_HOP_SECONDS = 0.04
DEFAULT_SHARD_SIZE = 128

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200"
    / "partial_as_fuss_e4_unfreeze1_highthr_resume"
    / "pretrainedsed_beats_qces_detector.pt"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/detector_manifest_val.jsonl"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/ontology_200_trainable.txt"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200"
    / "partial_as_fuss_e4_unfreeze1_highthr_resume"
    / "val_dense_features_v2"
)


@dataclass(frozen=True)
class ExportRow:
    """Detector row plus provenance fields discarded by the training loader."""

    scene: SceneItem
    source_route: str
    raw: Mapping[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--audio-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-name", default="BEATs_strong_1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--max-scenes", type=int, default=0, help="Limit after all filters; 0 keeps every scene.")
    parser.add_argument(
        "--source-route",
        action="append",
        default=[],
        help="Keep an exact source_route value. Repeat to select multiple routes.",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        help="Optional .txt/.json/.jsonl scene-id allowlist.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="Additional scene id to keep. Repeat as needed.",
    )
    parser.add_argument("--include-probs", action="store_true", help="Also store sigmoid(logits) as float16.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_scene_ids(path: Path) -> set[str]:
    """Read scene ids from a JSON list/dict, JSONL, or one-id-per-line file."""

    path = path.resolve()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict) and isinstance(payload.get("scene_ids"), list):
            values = payload["scene_ids"]
        else:
            raise ValueError(f"JSON scene list must be a list or contain scene_ids: {path}")
        result = {str(value).strip() for value in values if str(value).strip()}
    elif path.suffix.lower() == ".jsonl":
        result: set[str] = set()
        for row in read_jsonl(path):
            scene_id = str(row.get("scene_id") or "").strip()
            if not scene_id:
                raise ValueError(f"JSONL row has no scene_id: {path}")
            result.add(scene_id)
    else:
        result = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    if not result:
        raise ValueError(f"empty scene list: {path}")
    return result


def load_filtered_rows(
    manifest: Path,
    label_to_id: Mapping[str, int],
    *,
    source_routes: set[str],
    scene_ids: set[str] | None,
    max_scenes: int,
) -> list[ExportRow]:
    rows: list[ExportRow] = []
    seen_scene_ids: set[str] = set()
    for raw in read_jsonl(manifest):
        scene_id = str(raw.get("scene_id") or "").strip()
        source_route = str(raw.get("source_route") or "").strip()
        if not scene_id:
            raise ValueError(f"manifest row has no scene_id: {manifest}")
        if scene_ids is not None and scene_id not in scene_ids:
            continue
        if source_routes and source_route not in source_routes:
            continue
        if scene_id in seen_scene_ids:
            raise ValueError(f"duplicate scene_id after filtering: {scene_id}")

        events: list[dict[str, Any]] = []
        for event in raw.get("events") or []:
            label = str(event.get("label") or "")
            if label not in label_to_id:
                continue
            events.append(
                {
                    **event,
                    "label": label,
                    "label_id": int(label_to_id[label]),
                    "onset_seconds": float(event.get("onset_seconds", 0.0)),
                    "offset_seconds": float(event.get("offset_seconds", 0.0)),
                }
            )
        scene = SceneItem(
            scene_id=scene_id,
            split=str(raw.get("split") or raw.get("hf_split") or ""),
            mixture_path=str(raw["mixture_path"]),
            duration_seconds=float(raw.get("duration_seconds") or FIXED_AUDIO_SECONDS),
            sample_rate=int(raw.get("sample_rate") or 32_000),
            events=tuple(events),
        )
        rows.append(ExportRow(scene=scene, source_route=source_route, raw=raw))
        seen_scene_ids.add(scene_id)
        if max_scenes > 0 and len(rows) >= max_scenes:
            break
    return rows


def valid_frames_for_duration(duration_seconds: float) -> int:
    """Number of non-padding frames on the fixed left-aligned 40 ms grid."""

    clipped = min(max(float(duration_seconds), 0.0), FIXED_AUDIO_SECONDS)
    # Durations are serialized as float32.  The tiny tolerance keeps an exact
    # grid boundary such as 9.96 s from becoming an extra frame after the
    # float32 round-trip, while remaining far below the manifest precision.
    return min(NUM_FRAMES, max(0, int(math.ceil(clipped / FRAME_HOP_SECONDS - 1e-5))))


@torch.inference_mode()
def extract_pre_head_features(model: torch.nn.Module, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the representation immediately before strong_head and its logits."""

    mel = model.mel_forward(waveforms)
    features = model.model(mel)
    features = interpolate_sequence(features, model.seq_len)
    features = model.seq_model(features)
    logits = model.strong_head(features)
    return features, logits


def fixed_grid_metadata() -> dict[str, Any]:
    return {
        "num_frames": NUM_FRAMES,
        "fixed_audio_seconds": FIXED_AUDIO_SECONDS,
        "frame_hop_seconds": FRAME_HOP_SECONDS,
        "frame_width_seconds": FRAME_HOP_SECONDS,
        "frame_interval_convention": "frame i covers [i*0.04, (i+1)*0.04)",
        "frame_timestamp_convention": "left_edge_seconds = i*0.04",
        "padding": "right",
        "truncation": "right_at_10_seconds",
    }


def validate_shard_payload(
    payload: Mapping[str, Any],
    *,
    num_labels: int,
    feature_dim: int,
    include_probs: bool,
) -> None:
    """Fail fast if a shard cannot be consumed as dense-feature format v2."""

    if payload.get("format") != SHARD_FORMAT:
        raise ValueError(f"bad shard format: {payload.get('format')!r}")
    scene_ids = payload.get("scene_ids")
    scenes = payload.get("scenes")
    features = payload.get("features")
    logits = payload.get("logits")
    durations = payload.get("duration_seconds")
    valid_frames = payload.get("valid_frames")
    gold_events = payload.get("gold_events")
    if not isinstance(scene_ids, list) or not scene_ids:
        raise ValueError("shard scene_ids must be a non-empty list")
    count = len(scene_ids)
    if len(set(scene_ids)) != count:
        raise ValueError("duplicate scene_ids inside shard")
    for name, value in (("scenes", scenes), ("gold_events", gold_events)):
        if not isinstance(value, list) or len(value) != count:
            raise ValueError(f"{name} must align with scene_ids")
    expected_feature_shape = (count, NUM_FRAMES, feature_dim)
    expected_logit_shape = (count, NUM_FRAMES, num_labels)
    if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_feature_shape:
        raise ValueError(f"features shape must be {expected_feature_shape}")
    if features.dtype != torch.float16:
        raise ValueError("features must be float16")
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected_logit_shape:
        raise ValueError(f"logits shape must be {expected_logit_shape}")
    if logits.dtype != torch.float16:
        raise ValueError("logits must be float16")
    if not isinstance(durations, torch.Tensor) or tuple(durations.shape) != (count,):
        raise ValueError("duration_seconds must be a one-dimensional aligned tensor")
    if not isinstance(valid_frames, torch.Tensor) or tuple(valid_frames.shape) != (count,):
        raise ValueError("valid_frames must be a one-dimensional aligned tensor")
    if valid_frames.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("valid_frames must use an integer dtype")
    expected_valid = torch.tensor(
        [valid_frames_for_duration(float(value)) for value in durations.tolist()],
        dtype=torch.int64,
    )
    if not torch.equal(valid_frames.to(torch.int64).cpu(), expected_valid):
        raise ValueError("valid_frames does not match the fixed 40 ms grid")
    probs = payload.get("probs")
    if include_probs:
        if not isinstance(probs, torch.Tensor) or tuple(probs.shape) != expected_logit_shape:
            raise ValueError(f"probs shape must be {expected_logit_shape}")
        if probs.dtype != torch.float16:
            raise ValueError("probs must be float16")
    elif "probs" in payload:
        raise ValueError("unexpected probs tensor when include_probs is false")


def validate_index_payload(payload: Mapping[str, Any]) -> None:
    """Validate the top-level schema and its fixed-grid invariants."""

    if payload.get("format") != FORMAT or payload.get("schema_version") != 2:
        raise ValueError("bad dense-feature index format/version")
    labels = payload.get("labels")
    shards = payload.get("shards")
    grid = payload.get("grid")
    tensors = payload.get("tensors")
    if not isinstance(labels, list) or len(labels) != int(payload.get("num_labels", -1)):
        raise ValueError("index labels/num_labels mismatch")
    if not isinstance(shards, list) or len(shards) != int(payload.get("shard_count", -1)):
        raise ValueError("index shards/shard_count mismatch")
    if sum(int(shard.get("scene_count", -1)) for shard in shards) != int(payload.get("scene_count", -1)):
        raise ValueError("index shard scene counts mismatch")
    if [int(shard.get("index", -1)) for shard in shards] != list(range(len(shards))):
        raise ValueError("index shard indices are not contiguous")
    shard_paths = [str(shard.get("path") or "") for shard in shards]
    if not all(shard_paths) or len(set(shard_paths)) != len(shard_paths):
        raise ValueError("index shard paths must be non-empty and unique")
    expected_grid = fixed_grid_metadata()
    if not isinstance(grid, Mapping) or any(grid.get(key) != value for key, value in expected_grid.items()):
        raise ValueError("index does not describe the fixed 250-frame/40-ms grid")
    if not isinstance(tensors, Mapping):
        raise ValueError("index tensors schema is missing")
    feature_dim = int(payload.get("feature_dim", -1))
    if tensors.get("features", {}).get("shape_per_scene") != [NUM_FRAMES, feature_dim]:
        raise ValueError("index feature shape mismatch")
    if tensors.get("logits", {}).get("shape_per_scene") != [NUM_FRAMES, len(labels)]:
        raise ValueError("index logit shape mismatch")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_shard_payload(
    entries: Sequence[dict[str, Any]],
    *,
    shard_index: int,
    run_id: str,
    include_probs: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": SHARD_FORMAT,
        "run_id": run_id,
        "shard_index": int(shard_index),
        "scene_ids": [str(entry["scene_id"]) for entry in entries],
        "scenes": [dict(entry["scene_metadata"]) for entry in entries],
        "features": torch.stack([entry["features"] for entry in entries], dim=0),
        "logits": torch.stack([entry["logits"] for entry in entries], dim=0),
        "duration_seconds": torch.tensor(
            [float(entry["duration_seconds"]) for entry in entries], dtype=torch.float32
        ),
        "valid_frames": torch.tensor([int(entry["valid_frames"]) for entry in entries], dtype=torch.int16),
        "gold_events": [list(entry["gold_events"]) for entry in entries],
    }
    if include_probs:
        payload["probs"] = torch.stack([entry["probs"] for entry in entries], dim=0)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive")
    if args.max_scenes < 0:
        raise SystemExit("--max-scenes must be non-negative")

    output_dir = args.output_dir.resolve()
    index_path = output_dir / "index.json"
    if index_path.exists() and not args.overwrite:
        raise SystemExit(f"output index exists: {index_path}; use --overwrite")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not index_path.exists():
        raise SystemExit(f"output dir is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_ontology(args.ontology.resolve())
    label_to_id = {label: index for index, label in enumerate(labels)}
    allowlist: set[str] | None = None
    if args.scene_list is not None:
        allowlist = load_scene_ids(args.scene_list)
    explicit_ids = {str(value).strip() for value in args.scene_id if str(value).strip()}
    if explicit_ids:
        allowlist = explicit_ids if allowlist is None else allowlist | explicit_ids
    source_routes = {str(value).strip() for value in args.source_route if str(value).strip()}
    export_rows = load_filtered_rows(
        args.manifest.resolve(),
        label_to_id,
        source_routes=source_routes,
        scene_ids=allowlist,
        max_scenes=args.max_scenes,
    )
    if not export_rows:
        raise SystemExit("empty manifest after ontology/source-route/scene-list filtering")

    device = make_device(args.device)
    checkpoint = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=False)
    checkpoint_labels = checkpoint.get("labels")
    if checkpoint_labels != labels:
        raise ValueError(
            f"checkpoint/ontology labels mismatch: checkpoint={len(checkpoint_labels or [])}, ontology={len(labels)}"
        )
    model = load_model(len(labels), args.checkpoint_name, device, unfreeze_last_blocks=0)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if int(model.seq_len) != NUM_FRAMES:
        raise ValueError(f"expected detector seq_len={NUM_FRAMES}, got {model.seq_len}")
    feature_dim = int(model.strong_head.in_features)
    if feature_dim != 768:
        raise ValueError(f"expected BEATs pre-head feature_dim=768, got {feature_dim}")

    scenes = [row.scene for row in export_rows]
    metadata_by_id = {row.scene.scene_id: row for row in export_rows}
    loader = DataLoader(
        SceneDataset(scenes, audio_root=args.audio_root.resolve(), fixed_seconds=FIXED_AUDIO_SECONDS),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    shard_descriptors: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    exported_scene_ids: list[str] = []

    def flush_shard() -> None:
        if not pending:
            return
        shard_index = len(shard_descriptors)
        shard_payload = _make_shard_payload(
            pending,
            shard_index=shard_index,
            run_id=run_id,
            include_probs=args.include_probs,
        )
        validate_shard_payload(
            shard_payload,
            num_labels=len(labels),
            feature_dim=feature_dim,
            include_probs=args.include_probs,
        )
        shard_name = f"shard-{run_id}-{shard_index:05d}.pt"
        shard_path = output_dir / shard_name
        _atomic_torch_save(shard_payload, shard_path)
        descriptor = {
            "index": shard_index,
            "path": shard_name,
            "scene_count": len(pending),
            "first_scene_id": str(pending[0]["scene_id"]),
            "last_scene_id": str(pending[-1]["scene_id"]),
            "bytes": shard_path.stat().st_size,
            "sha256": _sha256_file(shard_path),
        }
        shard_descriptors.append(descriptor)
        print(
            f"wrote shard {shard_index + 1}: scenes={len(pending)} total={len(exported_scene_ids)} "
            f"path={shard_name}",
            flush=True,
        )
        pending.clear()

    for waveforms, batch_rows in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        features, logits = extract_pre_head_features(model, waveforms)
        if features.ndim != 3 or logits.ndim != 3:
            raise ValueError(f"unexpected dense tensor ranks: features={features.shape}, logits={logits.shape}")
        if features.shape[1:] != (NUM_FRAMES, feature_dim):
            raise ValueError(f"unexpected features shape: {tuple(features.shape)}")
        if logits.shape[1:] != (NUM_FRAMES, len(labels)):
            raise ValueError(f"unexpected logits shape: {tuple(logits.shape)}")
        features_cpu = features.detach().to(device="cpu", dtype=torch.float16)
        logits_cpu = logits.detach().to(device="cpu", dtype=torch.float16)
        probs_cpu = (
            torch.sigmoid(logits).detach().to(device="cpu", dtype=torch.float16)
            if args.include_probs
            else None
        )
        for batch_index, scene in enumerate(batch_rows):
            export_row = metadata_by_id[scene.scene_id]
            valid_frames = valid_frames_for_duration(scene.duration_seconds)
            scene_metadata = {
                "scene_id": scene.scene_id,
                "split": scene.split,
                "source_route": export_row.source_route,
                "mixture_path": scene.mixture_path,
                "duration_seconds": float(scene.duration_seconds),
                "sample_rate": int(scene.sample_rate),
                "valid_frames": valid_frames,
                "video_id": export_row.raw.get("video_id"),
                "audio_sha256": export_row.raw.get("audio_sha256"),
            }
            entry: dict[str, Any] = {
                "scene_id": scene.scene_id,
                "scene_metadata": scene_metadata,
                "features": features_cpu[batch_index].contiguous(),
                "logits": logits_cpu[batch_index].contiguous(),
                "duration_seconds": float(scene.duration_seconds),
                "valid_frames": valid_frames,
                "gold_events": list(scene.events),
            }
            if probs_cpu is not None:
                entry["probs"] = probs_cpu[batch_index].contiguous()
            pending.append(entry)
            exported_scene_ids.append(scene.scene_id)
            if len(pending) >= args.shard_size:
                flush_shard()
    flush_shard()

    if exported_scene_ids != [row.scene.scene_id for row in export_rows]:
        raise RuntimeError("export order/count differs from filtered manifest")
    if sum(int(shard["scene_count"]) for shard in shard_descriptors) != len(export_rows):
        raise RuntimeError("shard scene counts do not sum to filtered manifest size")

    index_payload = {
        "format": FORMAT,
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "detector_checkpoint": str(args.detector_checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "ontology": str(args.ontology.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "labels": labels,
        "num_labels": len(labels),
        "feature_dim": feature_dim,
        "scene_count": len(export_rows),
        "shard_count": len(shard_descriptors),
        "shard_size_requested": int(args.shard_size),
        "grid": fixed_grid_metadata(),
        "tensors": {
            "features": {"shape_per_scene": [NUM_FRAMES, feature_dim], "dtype": "torch.float16", "pre_head": True},
            "logits": {"shape_per_scene": [NUM_FRAMES, len(labels)], "dtype": "torch.float16"},
            "probs": (
                {"shape_per_scene": [NUM_FRAMES, len(labels)], "dtype": "torch.float16"}
                if args.include_probs
                else None
            ),
            "duration_seconds": {"shape_per_scene": [], "dtype": "torch.float32"},
            "valid_frames": {"shape_per_scene": [], "dtype": "torch.int16"},
        },
        "include_probs": bool(args.include_probs),
        "filters": {
            "source_routes": sorted(source_routes),
            "scene_list": str(args.scene_list.resolve()) if args.scene_list is not None else None,
            "explicit_scene_ids": sorted(explicit_ids),
            "allowlist_size": len(allowlist) if allowlist is not None else None,
            "max_scenes_after_filter": int(args.max_scenes),
        },
        "shards": shard_descriptors,
    }
    validate_index_payload(index_payload)
    _atomic_json_save(index_payload, index_path)
    print(json.dumps(index_payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
