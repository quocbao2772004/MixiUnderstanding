#!/usr/bin/env python3
"""Evaluate frozen text-only AudioSep followed by predicted temporal crop.

This is the actuator that matches the decoupled design: AudioSep receives only
an on-manifold canonical text prompt and a local audio window.  The detector
span is never an AudioSep input; it is applied to the returned waveform.  The
target is used only for offline metrics and oracle candidate headroom.
"""

from __future__ import annotations

import argparse
import gc
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
import torchaudio.functional as AF
from torch.utils.data import DataLoader

from mixi_understanding.qces.metrics import scale_dependent_sdr
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    encode_prompts,
)
from mixi_understanding.scripts.train_qces_public22_robust_separator_v1 import (
    RobustConditionDataset,
    _read_jsonl,
    _stratified_limit,
)
from mixi_understanding.scripts.train_qces_span_conditioned_separator_v1 import RATE


FORMAT = "qces_public22_audiosep_semantic_postcrop_v1"
AUDIOSEP_RATE = 32_000
PROJECTION_BLEND_ALPHAS = tuple(index / 10.0 for index in range(1, 10))
PROMPTS = {
    "Bark": "a dog barking",
    "Howl": "a dog howling",
    "Whimper_(dog)": "a dog whimpering",
    "Meow": "a cat meowing",
    "Purr": "a cat purring",
    "Caterwaul": "a cat yowling",
    "Clapping": "people clapping",
    "Laughter": "a person laughing",
    "Giggle": "a person giggling",
    "Conversation": "people having a conversation",
    "Shout": "a person shouting",
    "Crying_and_sobbing": "a person crying and sobbing",
    "Knock": "a knocking sound",
    "Slam": "a slamming sound",
    "Thump_and_thud": "a thumping and thudding sound",
    "Traffic_noise_and_roadway_noise": "road traffic noise",
    "Heavy_engine_(low_frequency)": "a heavy low-frequency engine",
    "Reversing_beeps": "vehicle reversing beeps",
    "Speech": "a person speaking",
    "Air_horn_and_truck_horn": "a loud truck air horn",
    "Police_car_(siren)": "a police car siren",
    "Engine_starting": "an engine starting",
}


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_public22_separator_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-manifest", type=Path, default=data / "event_components_dev.jsonl")
    parser.add_argument("--audiosep-root", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep")
    parser.add_argument(
        "--audiosep-config", type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep/config/audiosep_base.yaml",
    )
    parser.add_argument(
        "--audiosep-checkpoint", type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_public22_audiosep_postcrop_v1",
    )
    parser.add_argument("--chunk-seconds", type=float, default=2.56)
    parser.add_argument("--minimum-condition-iou", type=float, default=0.30)
    parser.add_argument("--validation-pad-seconds", type=float, default=0.15)
    parser.add_argument("--soft-edge-seconds", type=float, default=0.04)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2253)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--projection-candidates", action=argparse.BooleanOptionalAction, default=False,
        help="Add target-free least-squares amplitude projection and conservative blends.",
    )
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


def _aggregate(rows: Sequence[Mapping[str, Any]], candidate_names: Sequence[str]) -> dict[str, Any]:
    crop = np.asarray([float(row["sd_sdr_db"]["crop"]) for row in rows], dtype=np.float64)
    result: dict[str, Any] = {"events": len(rows), "crop_sd_sdr_db_↑": _summary(crop.tolist())}
    for name in candidate_names:
        scores = np.asarray([float(row["sd_sdr_db"][name]) for row in rows], dtype=np.float64)
        gains = scores - crop
        result[name] = {
            "sd_sdr_db_↑": _summary(scores.tolist()),
            "gain_over_crop_db_↑": _summary(gains.tolist()),
            "positive_rate_↑": float(np.mean(gains > 0.0)),
            "harmful_below_minus_1db_rate_↓": float(np.mean(gains < -1.0)),
        }
    oracle = np.asarray([float(row["oracle_score_db"]) for row in rows], dtype=np.float64)
    oracle_gain = oracle - crop
    result["oracle_selector"] = {
        "sd_sdr_db_↑": _summary(oracle.tolist()),
        "gain_over_crop_db_↑": _summary(oracle_gain.tolist()),
        "positive_rate_↑": float(np.mean(oracle_gain > 1e-7)),
        "gain_at_least_1db_rate_↑": float(np.mean(oracle_gain >= 1.0)),
        "gain_at_least_2db_rate_↑": float(np.mean(oracle_gain >= 2.0)),
        "selection_counts": {
            name: sum(str(row["oracle_candidate"]) == name for row in rows)
            for name in ["crop", *candidate_names]
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
    rows = [
        row for row in _read_jsonl(args.dev_manifest.resolve())
        if bool(row.get("condition_available"))
        and float(row["condition_iou"]) >= args.minimum_condition_iou
    ]
    rows = _stratified_limit(rows, args.max_events, args.seed)
    labels = sorted({str(row["label"]) for row in rows})
    missing_prompts = sorted(set(labels) - set(PROMPTS))
    if missing_prompts:
        raise ValueError(f"missing canonical prompts for: {missing_prompts}")

    prompt_embeddings = encode_prompts(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(),
        [PROMPTS[label] for label in labels], batch_size=22,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    separator = _load_separator(args, device)
    dataset = RobustConditionDataset(
        rows,
        chunk_samples=int(round(args.chunk_seconds * RATE)),
        seed=args.seed + 1,
        train=False,
        exact_probability=0.0,
        predicted_probability=1.0,
        pad_min=0,
        pad_max=0,
        validation_pad=int(round(args.validation_pad_seconds * RATE)),
        maximum_jitter=0,
        soft_edge=int(round(args.soft_edge_seconds * RATE)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    candidate_names = [
        "audiosep_raw", "audiosep_postcrop",
        "audiosep_blend_0.25", "audiosep_blend_0.50", "audiosep_blend_0.75",
    ]
    if args.projection_candidates:
        candidate_names.extend([
            "audiosep_projected",
            *(f"projected_blend_{alpha:.2f}" for alpha in PROJECTION_BLEND_ALPHAS),
        ])
    items: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        mixture, target, condition, _gold, condition_iou, label_id, proxy, _mode = batch
        label = str(rows[index]["label"])
        mixture = mixture.to(device)
        target = target.to(device)
        condition = condition.to(device)
        mixture_32k = AF.resample(mixture, RATE, AUDIOSEP_RATE)
        embedding = prompt_embeddings[PROMPTS[label]][None].to(device)
        raw_32k = separator({"mixture": mixture_32k[:, None], "condition": embedding})["waveform"][:, 0]
        raw = AF.resample(raw_32k, AUDIOSEP_RATE, RATE)
        if raw.shape[-1] != mixture.shape[-1]:
            raw = torch.nn.functional.interpolate(
                raw[:, None], size=mixture.shape[-1], mode="linear", align_corners=False
            )[:, 0]
        crop = mixture * condition
        postcrop = raw * condition
        candidates: dict[str, torch.Tensor] = {
            "crop": crop,
            "audiosep_raw": raw,
            "audiosep_postcrop": postcrop,
        }
        for alpha in (0.25, 0.50, 0.75):
            candidates[f"audiosep_blend_{alpha:.2f}"] = (1.0 - alpha) * crop + alpha * postcrop
        projection_scale = torch.ones(mixture.shape[0], 1, device=device)
        if args.projection_candidates:
            # Target-free least-squares scale of the separated candidate onto
            # the observed crop.  Clamping prevents an uncertain, low-energy
            # output from being amplified without bound.
            projection_scale = (
                (postcrop * crop).sum(-1, keepdim=True)
                / postcrop.square().sum(-1, keepdim=True).clamp_min(1e-8)
            ).clamp(0.25, 4.0)
            projected = postcrop * projection_scale
            candidates["audiosep_projected"] = projected
            for alpha in PROJECTION_BLEND_ALPHAS:
                candidates[f"projected_blend_{alpha:.2f}"] = (
                    (1.0 - alpha) * crop + alpha * projected
                )
        scores = {
            name: float(scale_dependent_sdr(value, target)[0].cpu())
            for name, value in candidates.items()
        }
        oracle_candidate = max(scores, key=scores.get)
        items.append({
            "event_id": str(rows[index]["event_id"]),
            "scene_id": str(rows[index]["scene_id"]),
            "source_id": str(rows[index]["source_id"]),
            "label": label,
            "prompt": PROMPTS[label],
            "semantic_supervision": str(rows[index].get("semantic_supervision", "unknown")),
            "is_proxy": bool(int(proxy[0])),
            "condition_iou": float(condition_iou[0]),
            "condition_score": rows[index].get("condition_score"),
            "sd_sdr_db": scores,
            "gain_over_crop_db": {
                name: float(scores[name] - scores["crop"]) for name in candidate_names
            },
            "oracle_candidate": oracle_candidate,
            "oracle_score_db": float(scores[oracle_candidate]),
            "oracle_gain_over_crop_db": float(scores[oracle_candidate] - scores["crop"]),
            "projection_scale": float(projection_scale[0, 0].cpu()),
        })
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(json.dumps({"done": index + 1, "total": len(rows)}), flush=True)

    del separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    exact = [row for row in items if not bool(row["is_proxy"])]
    proxy = [row for row in items if bool(row["is_proxy"])]
    aggregate = {
        "all": _aggregate(items, candidate_names),
        "exact_semantics": _aggregate(exact, candidate_names),
        "proxy_semantics": _aggregate(proxy, candidate_names),
        "per_label": {
            label: _aggregate([row for row in items if row["label"] == label], candidate_names)
            for label in labels
        },
    }
    oracle = aggregate["exact_semantics"]["oracle_selector"]
    passed = (
        float(oracle["gain_over_crop_db_↑"]["median"]) >= 2.0
        and float(oracle["positive_rate_↑"]) >= 0.75
    )
    items_path = output / "items.jsonl"
    _write_jsonl(items_path, items)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "events": len(items),
        "exact_events": len(exact),
        "proxy_events": len(proxy),
        "canonical_prompts": PROMPTS,
        "aggregate": aggregate,
        "go_no_go_gate": {
            "scope": "exact_semantics",
            "oracle_median_gain_db_min": 2.0,
            "oracle_positive_rate_min": 0.75,
            "passed": passed,
            "decision": "train_quality_selector" if passed else "audiosep_candidates_insufficient",
        },
        "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
        "audiosep_checkpoint_sha256": _sha256(args.audiosep_checkpoint.resolve()),
        "dev_manifest": str(args.dev_manifest.resolve()),
        "dev_manifest_sha256": _sha256(args.dev_manifest.resolve()),
        "items": str(items_path.resolve()),
        "items_sha256": _sha256(items_path),
        "protocol": {
            "audiosep_input_rate": AUDIOSEP_RATE,
            "metric_rate": RATE,
            "text_condition_only": True,
            "predicted_span_used_only_as_postcrop": True,
            "target_used_only_for_offline_metrics": True,
            "projection_candidates": bool(args.projection_candidates),
            "projection_scale_clip": [0.25, 4.0] if args.projection_candidates else None,
        },
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "complete": True,
        "events": len(items),
        "exact_audiosep_postcrop": aggregate["exact_semantics"]["audiosep_postcrop"],
        "exact_oracle": oracle,
        "gate": receipt["go_no_go_gate"],
        "receipt": str((output / "receipt.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
