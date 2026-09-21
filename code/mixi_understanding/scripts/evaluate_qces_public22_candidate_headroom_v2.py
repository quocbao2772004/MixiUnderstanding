#!/usr/bin/env python3
"""Audit deployable evidence candidates before training a quality selector.

The current public-22 separator is left untouched.  For every development
event this script evaluates deterministic candidates that do not use the
target at inference time:

* the padded detector crop;
* the current span-conditioned separator output;
* a post-cropped version of that output;
* semantic-only separation (all-one separator condition) followed by the
  detector crop; and
* conservative mixtures of crop and separated audio.

The target is used only by this offline evaluator to compute SD-SDR and the
oracle candidate ceiling.  Per-item results are persisted so a later selector
can be trained without re-running separation.
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
from torch.utils.data import DataLoader

from mixi_understanding.qces.metrics import scale_dependent_sdr
from mixi_understanding.scripts.train_qces_public22_robust_separator_v1 import (
    RobustConditionDataset,
    _read_jsonl,
)
from mixi_understanding.scripts.train_qces_span_conditioned_separator_v1 import (
    CHECKPOINT_FORMAT,
    RATE,
    SpanMaskNetwork,
    _separator_forward,
)


FORMAT = "qces_public22_candidate_headroom_v2"


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_public22_separator_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-manifest", type=Path, default=data / "event_components_dev.jsonl")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "outputs/qces_public22_robust_separator_v1_full/best.pt",
    )
    parser.add_argument(
        "--initial-label-cache", type=Path,
        default=PROJECT_ROOT / "outputs/qces_label_clap191_v1.pt",
    )
    parser.add_argument(
        "--label-embedding-cache", type=Path,
        default=PROJECT_ROOT / "outputs/qces_label_clap_public22_v1.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_public22_candidate_headroom_v2",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-seconds", type=float, default=2.56)
    parser.add_argument("--minimum-condition-iou", type=float, default=0.30)
    parser.add_argument("--validation-pad-seconds", type=float, default=0.15)
    parser.add_argument("--soft-edge-seconds", type=float, default=0.04)
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"mean": float("nan"), "median": float("nan"), "q10": float("nan"), "q90": float("nan")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], candidates: Sequence[str]) -> dict[str, Any]:
    crop = np.asarray([float(row["sd_sdr_db"]["crop"]) for row in rows], dtype=np.float64)
    result: dict[str, Any] = {"events": len(rows), "crop_sd_sdr_db_↑": _summary(crop.tolist())}
    for candidate in candidates:
        score = np.asarray([float(row["sd_sdr_db"][candidate]) for row in rows], dtype=np.float64)
        gain = score - crop
        result[candidate] = {
            "sd_sdr_db_↑": _summary(score.tolist()),
            "gain_over_crop_db_↑": _summary(gain.tolist()),
            "positive_rate_↑": float(np.mean(gain > 0.0)),
            "harmful_below_minus_1db_rate_↓": float(np.mean(gain < -1.0)),
        }
    oracle_score = np.asarray([float(row["oracle_score_db"]) for row in rows], dtype=np.float64)
    oracle_gain = oracle_score - crop
    result["oracle_selector"] = {
        "sd_sdr_db_↑": _summary(oracle_score.tolist()),
        "gain_over_crop_db_↑": _summary(oracle_gain.tolist()),
        "positive_rate_↑": float(np.mean(oracle_gain > 1e-7)),
        "gain_at_least_1db_rate_↑": float(np.mean(oracle_gain >= 1.0)),
        "gain_at_least_2db_rate_↑": float(np.mean(oracle_gain >= 2.0)),
        "selection_counts": {
            name: sum(str(row["oracle_candidate"]) == name for row in rows)
            for name in ["crop", *candidates]
        },
    }
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    all_rows = _read_jsonl(args.dev_manifest.resolve())
    rows = [
        row for row in all_rows
        if bool(row.get("condition_available"))
        and float(row["condition_iou"]) >= args.minimum_condition_iou
    ]
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("invalid separator checkpoint")
    initial_cache = torch.load(args.initial_label_cache.resolve(), map_location="cpu", weights_only=True)
    public_cache = torch.load(args.label_embedding_cache.resolve(), map_location="cpu", weights_only=True)
    labels = list(public_cache["labels"])
    config = checkpoint["config"]
    model = SpanMaskNetwork(
        base_channels=int(config["base_channels"]),
        semantic_channels=int(config["semantic_channels"]),
        label_embeddings=initial_cache["embeddings"].float(),
    )
    # SpanMaskNetwork's legacy constructor validates a 191-row initialization
    # bank.  Public-22 checkpoints correctly persist a 22-row buffer, so load
    # the trainable weights independently and attach the public bank below.
    model_state = dict(checkpoint["model_state"])
    stored_embeddings = model_state.pop("label_embeddings")
    if not torch.equal(stored_embeddings.cpu(), public_cache["embeddings"].float().cpu()):
        raise ValueError("checkpoint label embeddings do not match the public-22 cache")
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if set(missing) != {"label_embeddings"} or unexpected:
        raise RuntimeError(f"unexpected checkpoint incompatibility: missing={missing}, unexpected={unexpected}")
    model.label_embeddings = public_cache["embeddings"].float().contiguous()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataset = RobustConditionDataset(
        rows,
        chunk_samples=int(round(args.chunk_seconds * RATE)),
        seed=2248,
        train=False,
        exact_probability=0.0,
        predicted_probability=1.0,
        pad_min=0,
        pad_max=0,
        validation_pad=int(round(args.validation_pad_seconds * RATE)),
        maximum_jitter=0,
        soft_edge=int(round(args.soft_edge_seconds * RATE)),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    candidate_names = [
        "span_raw",
        "span_postcrop",
        "semantic_postcrop",
        "span_blend_0.25",
        "span_blend_0.50",
        "span_blend_0.75",
        "semantic_blend_0.25",
        "semantic_blend_0.50",
        "semantic_blend_0.75",
    ]
    item_rows: list[dict[str, Any]] = []
    cursor = 0
    for batch in loader:
        mixture, target, condition, _gold, condition_iou, label_id, proxy, _mode = batch
        mixture = mixture.to(device)
        target = target.to(device)
        condition_gpu = condition.to(device)
        label_gpu = label_id.to(device)
        span_raw, _, _, _ = _separator_forward(model, mixture, condition_gpu, label_gpu)
        semantic_full, _, _, _ = _separator_forward(
            model, mixture, torch.ones_like(condition_gpu), label_gpu
        )
        crop = mixture * condition_gpu
        span_postcrop = span_raw * condition_gpu
        semantic_postcrop = semantic_full * condition_gpu
        candidates: dict[str, torch.Tensor] = {
            "crop": crop,
            "span_raw": span_raw,
            "span_postcrop": span_postcrop,
            "semantic_postcrop": semantic_postcrop,
        }
        for alpha in (0.25, 0.50, 0.75):
            candidates[f"span_blend_{alpha:.2f}"] = (1.0 - alpha) * crop + alpha * span_postcrop
            candidates[f"semantic_blend_{alpha:.2f}"] = (1.0 - alpha) * crop + alpha * semantic_postcrop
        scores = {
            name: scale_dependent_sdr(value, target).detach().cpu().tolist()
            for name, value in candidates.items()
        }
        energy = {
            name: (
                value.square().sum(-1) / mixture.square().sum(-1).clamp_min(1e-8)
            ).detach().cpu().tolist()
            for name, value in candidates.items()
        }
        for local_index in range(mixture.shape[0]):
            source = rows[cursor]
            per_scores = {name: float(values[local_index]) for name, values in scores.items()}
            oracle_candidate = max(per_scores, key=per_scores.get)
            item_rows.append({
                "event_id": str(source["event_id"]),
                "scene_id": str(source["scene_id"]),
                "source_id": str(source["source_id"]),
                "label": labels[int(label_id[local_index])],
                "label_id": int(label_id[local_index]),
                "semantic_supervision": str(source.get("semantic_supervision", "unknown")),
                "condition_iou": float(condition_iou[local_index]),
                "condition_score": (
                    float(source["condition_score"])
                    if source.get("condition_score") is not None else float("nan")
                ),
                "sd_sdr_db": per_scores,
                "gain_over_crop_db": {
                    name: float(per_scores[name] - per_scores["crop"])
                    for name in candidate_names
                },
                "output_to_mixture_energy_ratio": {
                    name: float(values[local_index]) for name, values in energy.items()
                },
                "is_proxy": bool(int(proxy[local_index])),
                "oracle_candidate": oracle_candidate,
                "oracle_score_db": float(per_scores[oracle_candidate]),
                "oracle_gain_over_crop_db": float(per_scores[oracle_candidate] - per_scores["crop"]),
            })
            cursor += 1

    if cursor != len(rows):
        raise RuntimeError(f"evaluated {cursor} rows but expected {len(rows)}")
    exact = [row for row in item_rows if not bool(row["is_proxy"])]
    proxy = [row for row in item_rows if bool(row["is_proxy"])]
    per_label = {
        label: _aggregate([row for row in item_rows if row["label"] == label], candidate_names)
        for label in labels
    }
    aggregate = {
        "all": _aggregate(item_rows, candidate_names),
        "exact_semantics": _aggregate(exact, candidate_names),
        "proxy_semantics": _aggregate(proxy, candidate_names),
        "per_label": per_label,
    }
    exact_oracle = aggregate["exact_semantics"]["oracle_selector"]
    headroom_pass = (
        float(exact_oracle["gain_over_crop_db_↑"]["median"]) >= 2.0
        and float(exact_oracle["positive_rate_↑"]) >= 0.75
    )
    items_path = output / "items.jsonl"
    _write_jsonl(items_path, item_rows)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "events": len(item_rows),
        "exact_events": len(exact),
        "proxy_events": len(proxy),
        "candidates": ["crop", *candidate_names],
        "aggregate": aggregate,
        "go_no_go_gate": {
            "scope": "exact_semantics",
            "oracle_median_gain_db_min": 2.0,
            "oracle_positive_rate_min": 0.75,
            "passed": headroom_pass,
            "decision": "train_quality_selector" if headroom_pass else "improve_candidate_generator_first",
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "dev_manifest": str(args.dev_manifest.resolve()),
        "dev_manifest_sha256": _sha256(args.dev_manifest.resolve()),
        "items": str(items_path.resolve()),
        "items_sha256": _sha256(items_path),
        "note": "Oracle uses target only for offline ceiling measurement; it is not deployable.",
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "complete": True,
        "events": len(item_rows),
        "exact_oracle": exact_oracle,
        "gate": receipt["go_no_go_gate"],
        "receipt": str((output / "receipt.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
