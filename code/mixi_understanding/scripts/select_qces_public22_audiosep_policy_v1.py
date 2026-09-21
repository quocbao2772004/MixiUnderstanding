#!/usr/bin/env python3
"""Lock an AudioSep refinement policy on train and evaluate it once on dev.

The global projection/blend candidate is selected exclusively from train
exact-semantics metrics.  Dev targets never influence the policy.  Labels
whose waveform supervision is only an acoustic proxy always use crop fallback.
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

import numpy as np


FORMAT = "qces_public22_audiosep_locked_policy_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-receipt", type=Path, required=True)
    parser.add_argument("--dev-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def _metrics(rows: Sequence[Mapping[str, Any]], candidate: str) -> dict[str, Any]:
    crop = np.asarray([float(row["sd_sdr_db"]["crop"]) for row in rows], dtype=np.float64)
    selected = np.asarray([
        float(row["sd_sdr_db"][candidate])
        if not bool(row["is_proxy"]) else float(row["sd_sdr_db"]["crop"])
        for row in rows
    ], dtype=np.float64)
    gain = selected - crop
    return {
        "events": len(rows),
        "candidate_for_exact_semantics": candidate,
        "proxy_policy": "crop_fallback",
        "selected_sd_sdr_db_↑": _summary(selected.tolist()),
        "crop_sd_sdr_db_↑": _summary(crop.tolist()),
        "gain_over_crop_db_↑": _summary(gain.tolist()),
        "positive_rate_↑": float(np.mean(gain > 0.0)),
        "nonnegative_rate_↑": float(np.mean(gain >= 0.0)),
        "harmful_below_minus_1db_rate_↓": float(np.mean(gain < -1.0)),
    }


def main() -> None:
    args = parse_args()
    train_path = args.train_receipt.resolve()
    dev_path = args.dev_receipt.resolve()
    train = json.loads(train_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if not train.get("complete") or not dev.get("complete"):
        raise ValueError("train/dev AudioSep receipts must be complete")
    if train["audiosep_checkpoint_sha256"] != dev["audiosep_checkpoint_sha256"]:
        raise ValueError("train/dev were produced by different AudioSep checkpoints")
    train_exact = train["aggregate"]["exact_semantics"]
    candidates = sorted(
        name for name, value in train_exact.items()
        if name.startswith("projected_blend_") and isinstance(value, dict)
    )
    if not candidates:
        raise ValueError("train receipt has no projection blend candidates")
    train_constraints = {
        "mean_gain_db_min": 0.0,
        "positive_rate_min": 0.60,
        "harmful_below_minus_1db_rate_max": 0.25,
    }
    eligible = []
    for candidate in candidates:
        metrics = train_exact[candidate]
        gain = metrics["gain_over_crop_db_↑"]
        if (
            float(gain["mean"]) >= train_constraints["mean_gain_db_min"]
            and float(metrics["positive_rate_↑"]) >= train_constraints["positive_rate_min"]
            and float(metrics["harmful_below_minus_1db_rate_↓"])
            <= train_constraints["harmful_below_minus_1db_rate_max"]
        ):
            eligible.append(candidate)
    if not eligible:
        raise RuntimeError("no projection blend satisfies the predeclared train safety constraints")
    selected_candidate = max(
        eligible,
        key=lambda candidate: (
            float(train_exact[candidate]["gain_over_crop_db_↑"]["median"]),
            float(train_exact[candidate]["positive_rate_↑"]),
            float(train_exact[candidate]["gain_over_crop_db_↑"]["mean"]),
            -float(train_exact[candidate]["harmful_below_minus_1db_rate_↓"]),
        ),
    )

    train_items_path = Path(train["items"])
    dev_items_path = Path(dev["items"])
    train_items = _read_jsonl(train_items_path)
    dev_items = _read_jsonl(dev_items_path)
    dev_exact = [row for row in dev_items if not bool(row["is_proxy"])]
    dev_proxy = [row for row in dev_items if bool(row["is_proxy"])]
    dev_metrics = {
        "all": _metrics(dev_items, selected_candidate),
        "exact_semantics": _metrics(dev_exact, selected_candidate),
        "proxy_semantics": _metrics(dev_proxy, selected_candidate),
        "per_label": {
            label: _metrics([row for row in dev_items if row["label"] == label], selected_candidate)
            for label in sorted({str(row["label"]) for row in dev_items})
        },
    }
    exact = dev_metrics["exact_semantics"]
    gain = exact["gain_over_crop_db_↑"]
    dev_gate = {
        "median_gain_db_min": 1.0,
        "mean_gain_db_min": 0.0,
        "positive_rate_min": 0.60,
        "harmful_below_minus_1db_rate_max": 0.15,
    }
    accepted = (
        float(gain["median"]) >= dev_gate["median_gain_db_min"]
        and float(gain["mean"]) >= dev_gate["mean_gain_db_min"]
        and float(exact["positive_rate_↑"]) >= dev_gate["positive_rate_min"]
        and float(exact["harmful_below_minus_1db_rate_↓"])
        <= dev_gate["harmful_below_minus_1db_rate_max"]
    )
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "selection_protocol": {
            "uses_dev_targets_for_selection": False,
            "candidate_family": "target-free_projection_then_observation_blend",
            "train_constraints": train_constraints,
            "eligible_candidates": eligible,
            "selected_candidate": selected_candidate,
            "selection_key": "maximum_train_exact_median_gain_then_positive_rate_then_mean_gain",
            "proxy_semantics": "crop_fallback",
        },
        "train_selected_candidate_metrics": train_exact[selected_candidate],
        "dev_metrics": dev_metrics,
        "dev_acceptance_gate": {**dev_gate, "passed": accepted},
        "decision": "candidate_for_public22_beta" if accepted else "keep_crop_fallback",
        "train_receipt": str(train_path),
        "train_receipt_sha256": _sha256(train_path),
        "dev_receipt": str(dev_path),
        "dev_receipt_sha256": _sha256(dev_path),
        "train_items_sha256": _sha256(train_items_path),
        "dev_items_sha256": _sha256(dev_items_path),
    }
    _atomic_json(args.output.resolve(), receipt)
    print(json.dumps({
        "selected_candidate": selected_candidate,
        "eligible_candidates": eligible,
        "train_metrics": train_exact[selected_candidate],
        "dev_exact_metrics": exact,
        "gate": receipt["dev_acceptance_gate"],
        "decision": receipt["decision"],
        "receipt": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
