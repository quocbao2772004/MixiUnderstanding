#!/usr/bin/env python3
"""Train a source-grouped, train-only selector for three evidence candidates.

Candidates are crop fallback, the custom span-postcrop separator, and frozen
AudioSep with target-free amplitude projection.  Two gain regressors are fit
on exact-semantics train events.  A safety threshold is calibrated with
source-grouped out-of-fold predictions; dev is evaluated exactly once.
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

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


FORMAT = "qces_public22_quality_selector_v1"
CANDIDATES = ("crop", "custom_span_postcrop", "audiosep_projected")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3] / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-custom-items", type=Path, default=root / "qces_public22_custom_candidates_v2_train/items.jsonl")
    parser.add_argument("--train-audiosep-items", type=Path, default=root / "qces_public22_audiosep_projection_grid_v3_train/items.jsonl")
    parser.add_argument("--dev-custom-items", type=Path, default=root / "qces_public22_candidate_headroom_v2/items.jsonl")
    parser.add_argument("--dev-audiosep-items", type=Path, default=root / "qces_public22_audiosep_projected_v2_full/items.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "qces_public22_quality_selector_v1")
    parser.add_argument("--seed", type=int, default=2257)
    return parser.parse_args()


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["event_id"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


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


def _feature_matrix(
    custom: Mapping[str, Mapping[str, Any]],
    audiosep: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    event_ids = sorted(set(custom) & set(audiosep))
    features: list[np.ndarray] = []
    gains: list[list[float]] = []
    groups: list[str] = []
    proxy: list[bool] = []
    for event_id in event_ids:
        custom_row = custom[event_id]
        audiosep_row = audiosep[event_id]
        if custom_row["label"] != audiosep_row["label"]:
            raise ValueError(f"label mismatch for {event_id}")
        vector = np.zeros(len(labels) + 8, dtype=np.float32)
        vector[label_to_id[str(custom_row["label"])]] = 1.0
        offset = len(labels)
        condition_score = float(custom_row.get("condition_score") or 0.0)
        vector[offset] = condition_score
        energy = custom_row["output_to_mixture_energy_ratio"]
        for position, name in enumerate(("crop", "span_raw", "span_postcrop", "semantic_postcrop"), 1):
            vector[offset + position] = np.log10(max(float(energy[name]), 1e-6))
        vector[offset + 5] = np.log10(max(float(audiosep_row.get("projection_scale", 1.0)), 1e-4))
        vector[offset + 6] = condition_score * condition_score
        vector[offset + 7] = 1.0
        crop_score = float(custom_row["sd_sdr_db"]["crop"])
        features.append(vector)
        gains.append([
            float(custom_row["sd_sdr_db"]["span_postcrop"]) - crop_score,
            float(audiosep_row["sd_sdr_db"]["audiosep_projected"]) - crop_score,
        ])
        groups.append(str(custom_row["source_id"]))
        proxy.append(bool(custom_row["is_proxy"]))
    return event_ids, np.asarray(features), np.asarray(gains), np.asarray(groups), np.asarray(proxy)


def _new_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=seed,
    )


def _policy_gains(predicted: np.ndarray, actual: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    choice = np.argmax(predicted, axis=1)
    confidence = predicted[np.arange(len(predicted)), choice]
    use_refinement = confidence > threshold
    selected_gain = np.where(use_refinement, actual[np.arange(len(actual)), choice], 0.0)
    selected_id = np.where(use_refinement, choice + 1, 0)
    return selected_gain, selected_id


def _gain_metrics(gains: np.ndarray, selected_id: np.ndarray) -> dict[str, Any]:
    return {
        "events": int(len(gains)),
        "gain_over_crop_db_↑": _summary(gains.tolist()),
        "positive_rate_↑": float(np.mean(gains > 0.0)),
        "nonnegative_rate_↑": float(np.mean(gains >= 0.0)),
        "harmful_below_minus_1db_rate_↓": float(np.mean(gains < -1.0)),
        "refinement_coverage_↑": float(np.mean(selected_id != 0)),
        "selection_counts": {
            CANDIDATES[index]: int(np.sum(selected_id == index))
            for index in range(len(CANDIDATES))
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_custom": args.train_custom_items.resolve(),
        "train_audiosep": args.train_audiosep_items.resolve(),
        "dev_custom": args.dev_custom_items.resolve(),
        "dev_audiosep": args.dev_audiosep_items.resolve(),
    }
    train_custom = _read_jsonl(paths["train_custom"])
    train_audiosep = _read_jsonl(paths["train_audiosep"])
    dev_custom = _read_jsonl(paths["dev_custom"])
    dev_audiosep = _read_jsonl(paths["dev_audiosep"])
    labels = sorted({str(row["label"]) for row in train_custom.values()})
    train_ids, train_x, train_y, train_groups, train_proxy = _feature_matrix(
        train_custom, train_audiosep, labels
    )
    dev_ids, dev_x, dev_y, _dev_groups, dev_proxy = _feature_matrix(
        dev_custom, dev_audiosep, labels
    )
    train_exact = ~train_proxy
    dev_exact = ~dev_proxy
    x = train_x[train_exact]
    y = train_y[train_exact]
    groups = train_groups[train_exact]
    fit_target = np.clip(y, -20.0, 20.0)

    oof = np.zeros_like(y)
    splitter = GroupKFold(n_splits=5)
    for fold, (fit_indices, held_indices) in enumerate(splitter.split(x, groups=groups)):
        for candidate_index in range(2):
            model = _new_regressor(args.seed + fold * 10 + candidate_index)
            model.fit(x[fit_indices], fit_target[fit_indices, candidate_index])
            oof[held_indices, candidate_index] = model.predict(x[held_indices])
    threshold_grid = (0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)
    threshold_metrics: dict[str, Any] = {}
    eligible_thresholds: list[float] = []
    for threshold in threshold_grid:
        gains, choices = _policy_gains(oof, y, threshold)
        metrics = _gain_metrics(gains, choices)
        threshold_metrics[f"{threshold:.2f}"] = metrics
        if float(metrics["harmful_below_minus_1db_rate_↓"]) <= 0.15:
            eligible_thresholds.append(threshold)
    if not eligible_thresholds:
        raise RuntimeError("no OOF threshold satisfies the 15% harmful-rate safety constraint")
    selected_threshold = max(
        eligible_thresholds,
        key=lambda threshold: (
            float(threshold_metrics[f"{threshold:.2f}"]["gain_over_crop_db_↑"]["mean"]),
            float(threshold_metrics[f"{threshold:.2f}"]["gain_over_crop_db_↑"]["median"]),
            float(threshold_metrics[f"{threshold:.2f}"]["positive_rate_↑"]),
        ),
    )

    models = []
    for candidate_index in range(2):
        model = _new_regressor(args.seed + candidate_index)
        model.fit(x, fit_target[:, candidate_index])
        models.append(model)
    dev_prediction = np.column_stack([model.predict(dev_x[dev_exact]) for model in models])
    dev_gains_exact, dev_choices_exact = _policy_gains(
        dev_prediction, dev_y[dev_exact], selected_threshold
    )
    all_gains = np.zeros(len(dev_ids), dtype=np.float64)
    all_choices = np.zeros(len(dev_ids), dtype=np.int64)
    all_gains[dev_exact] = dev_gains_exact
    all_choices[dev_exact] = dev_choices_exact
    exact_metrics = _gain_metrics(dev_gains_exact, dev_choices_exact)
    all_metrics = _gain_metrics(all_gains, all_choices)
    proxy_metrics = _gain_metrics(all_gains[dev_proxy], all_choices[dev_proxy])
    per_label = {}
    dev_labels = np.asarray([str(dev_custom[event_id]["label"]) for event_id in dev_ids])
    for label in labels:
        mask = dev_labels == label
        per_label[label] = _gain_metrics(all_gains[mask], all_choices[mask])

    gate = {
        "median_gain_db_min": 1.0,
        "mean_gain_db_min": 0.0,
        "positive_rate_min": 0.60,
        "harmful_below_minus_1db_rate_max": 0.15,
    }
    gain = exact_metrics["gain_over_crop_db_↑"]
    accepted = (
        float(gain["median"]) >= gate["median_gain_db_min"]
        and float(gain["mean"]) >= gate["mean_gain_db_min"]
        and float(exact_metrics["positive_rate_↑"]) >= gate["positive_rate_min"]
        and float(exact_metrics["harmful_below_minus_1db_rate_↓"])
        <= gate["harmful_below_minus_1db_rate_max"]
    )
    model_path = output / "selector.joblib"
    joblib.dump({
        "format": FORMAT,
        "labels": labels,
        "feature_count": int(train_x.shape[1]),
        "candidates": CANDIDATES,
        "threshold": selected_threshold,
        "models": models,
    }, model_path)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "candidates": CANDIDATES,
        "features": {
            "uses_target_at_inference": False,
            "uses_condition_iou": False,
            "label_one_hot": len(labels),
            "numeric": [
                "condition_score", "log_crop_energy_ratio", "log_custom_raw_energy_ratio",
                "log_custom_postcrop_energy_ratio", "log_semantic_postcrop_energy_ratio",
                "log_audiosep_projection_scale", "condition_score_squared", "bias",
            ],
        },
        "train": {
            "exact_events": int(np.sum(train_exact)),
            "proxy_events_excluded": int(np.sum(train_proxy)),
            "unique_source_groups": int(len(set(groups.tolist()))),
            "target_gain_clip_db": [-20.0, 20.0],
            "oof_folds_grouped_by_source": 5,
            "threshold_grid": threshold_grid,
            "threshold_metrics": threshold_metrics,
            "selected_threshold": selected_threshold,
            "threshold_selection": "maximum OOF mean gain subject to harmful rate <= 0.15",
        },
        "dev": {
            "all": all_metrics,
            "exact_semantics": exact_metrics,
            "proxy_semantics": proxy_metrics,
            "per_label": per_label,
        },
        "dev_acceptance_gate": {**gate, "passed": accepted},
        "decision": "candidate_for_public22_beta" if accepted else "keep_crop_fallback",
        "selector": str(model_path.resolve()),
        "selector_sha256": _sha256(model_path),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
        },
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "selected_threshold": selected_threshold,
        "oof": threshold_metrics[f"{selected_threshold:.2f}"],
        "dev_exact": exact_metrics,
        "gate": receipt["dev_acceptance_gate"],
        "decision": receipt["decision"],
        "receipt": str((output / "receipt.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
