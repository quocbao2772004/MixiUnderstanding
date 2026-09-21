#!/usr/bin/env python3
"""Cache crop, projected AudioSep, and clean target waveforms for TF refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import DataLoader

from mixi_understanding.qces.metrics import scale_dependent_sdr
from mixi_understanding.scripts.evaluate_audiosep_baselines import _load_separator, encode_prompts
from mixi_understanding.scripts.evaluate_qces_public22_audiosep_postcrop_v1 import (
    AUDIOSEP_RATE,
    PROMPTS,
)
from mixi_understanding.scripts.train_qces_public22_robust_separator_v1 import (
    RobustConditionDataset,
    _read_jsonl,
    _stratified_limit,
)
from mixi_understanding.scripts.train_qces_span_conditioned_separator_v1 import RATE


FORMAT = "qces_public22_audiosep_waveform_cache_v1"


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_public22_separator_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=data / "event_components_train.jsonl")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--audiosep-root", type=Path, default=CODE_ROOT / "baseline/audiosep")
    parser.add_argument(
        "--audiosep-config", type=Path,
        default=CODE_ROOT / "baseline/audiosep/config/audiosep_base.yaml",
    )
    parser.add_argument(
        "--audiosep-checkpoint", type=Path,
        default=CODE_ROOT / "baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_public22_audiosep_waveforms_v1_train",
    )
    parser.add_argument("--chunk-seconds", type=float, default=2.56)
    parser.add_argument("--minimum-condition-iou", type=float, default=0.30)
    parser.add_argument("--validation-pad-seconds", type=float, default=0.15)
    parser.add_argument("--soft-edge-seconds", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2265)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


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


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in _read_jsonl(args.manifest.resolve())
        if bool(row.get("condition_available"))
        and float(row["condition_iou"]) >= args.minimum_condition_iou
        and str(row.get("semantic_supervision")) == "exact"
    ]
    rows = _stratified_limit(rows, args.max_events, args.seed)
    labels = sorted({str(row["label"]) for row in rows})
    missing = sorted(set(labels) - set(PROMPTS))
    if missing:
        raise ValueError(f"missing prompts: {missing}")
    prompt_embeddings = encode_prompts(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(),
        [PROMPTS[label] for label in labels], batch_size=len(labels),
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    separator = _load_separator(
        SimpleNamespace(
            audiosep_root=args.audiosep_root,
            audiosep_config=args.audiosep_config,
            audiosep_checkpoint=args.audiosep_checkpoint,
        ),
        device,
    )
    dataset = RobustConditionDataset(
        rows, chunk_samples=int(round(args.chunk_seconds * RATE)), seed=args.seed,
        train=False, exact_probability=0.0, predicted_probability=1.0,
        pad_min=0, pad_max=0,
        validation_pad=int(round(args.validation_pad_seconds * RATE)),
        maximum_jitter=0, soft_edge=int(round(args.soft_edge_seconds * RATE)),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    crop_values: list[torch.Tensor] = []
    audiosep_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    label_values: list[torch.Tensor] = []
    projection_values: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    cursor = 0
    for mixture, target, condition, _gold, _iou, label_id, _proxy, _mode in loader:
        mixture = mixture.to(device)
        condition = condition.to(device)
        crop = mixture * condition
        batch_labels = [str(rows[cursor + index]["label"]) for index in range(len(mixture))]
        embedding = torch.stack([prompt_embeddings[PROMPTS[label]] for label in batch_labels]).to(device)
        mixture_32k = AF.resample(mixture, RATE, AUDIOSEP_RATE)
        raw_32k = separator({"mixture": mixture_32k[:, None], "condition": embedding})["waveform"][:, 0]
        raw = AF.resample(raw_32k, AUDIOSEP_RATE, RATE)
        if raw.shape[-1] != mixture.shape[-1]:
            raw = F.interpolate(raw[:, None], size=mixture.shape[-1], mode="linear", align_corners=False)[:, 0]
        postcrop = raw * condition
        projection = (
            (postcrop * crop).sum(-1, keepdim=True)
            / postcrop.square().sum(-1, keepdim=True).clamp_min(1e-8)
        ).clamp(0.25, 4.0)
        projected = postcrop * projection
        crop_values.append(crop.half().cpu())
        audiosep_values.append(projected.half().cpu())
        target_values.append(target.half().cpu())
        label_values.append(label_id.short().cpu())
        projection_values.append(projection[:, 0].float().cpu())
        crop_sdr = scale_dependent_sdr(crop, target.to(device)).float().cpu()
        audiosep_sdr = scale_dependent_sdr(projected, target.to(device)).float().cpu()
        for index, label in enumerate(batch_labels):
            row = rows[cursor + index]
            metadata.append({
                "event_id": str(row["event_id"]),
                "source_id": str(row["source_id"]),
                "scene_id": str(row["scene_id"]),
                "label": label,
                "crop_sd_sdr_db": float(crop_sdr[index]),
                "audiosep_sd_sdr_db": float(audiosep_sdr[index]),
            })
        cursor += len(mixture)
        if cursor % 25 < len(mixture) or cursor == len(rows):
            print(json.dumps({"done": cursor, "total": len(rows)}), flush=True)
    payload = {
        "format": FORMAT,
        "sample_rate": RATE,
        "labels": labels,
        "crop": torch.cat(crop_values),
        "audiosep_projected": torch.cat(audiosep_values),
        "target": torch.cat(target_values),
        "label_id": torch.cat(label_values),
        "projection_scale": torch.cat(projection_values),
        "metadata": metadata,
    }
    cache_path = output / "waveforms.pt"
    torch.save(payload, cache_path)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "split": args.split,
        "events": len(rows),
        "sample_rate": RATE,
        "samples_per_event": int(payload["crop"].shape[-1]),
        "waveform_dtype": str(payload["crop"].dtype),
        "exact_semantics_only": True,
        "target_use": "training/evaluation only; never a deployment input",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest.resolve()),
        "audiosep_checkpoint_sha256": _sha256(args.audiosep_checkpoint.resolve()),
        "cache": str(cache_path.resolve()),
        "cache_sha256": _sha256(cache_path),
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
