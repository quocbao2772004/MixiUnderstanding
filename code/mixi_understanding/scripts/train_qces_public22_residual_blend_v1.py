#!/usr/bin/env python3
"""Train a source-grouped identity-preserving AudioSep residual blend.

The deployable output is

    evidence = crop + alpha * (AudioSep_projected - crop)

where alpha is selected from a conservative 0.1--0.5 grid, and alpha=0 is the
identity fallback.  Models see only label and target-free inference features.
Clean targets are used solely to label candidate gains during training and to
evaluate the locked policy.  The OOF safety constraint uses a 95% Wilson upper
bound rather than a point estimate at the deployment boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


FORMAT = "qces_public22_identity_residual_blend_v1"
ALPHAS = (0.10, 0.20, 0.30, 0.40, 0.50)
RISK_COST_DB = 5.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3] / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-features", type=Path,
        default=root / "qces_public22_selector_features_v2_train/items.jsonl",
    )
    parser.add_argument(
        "--dev-features", type=Path,
        default=root / "qces_public22_selector_features_v2_dev/items.jsonl",
    )
    parser.add_argument(
        "--train-grid", type=Path,
        default=root / "qces_public22_audiosep_projection_grid_v3_train/items.jsonl",
    )
    parser.add_argument(
        "--dev-grid", type=Path,
        default=root / "qces_public22_audiosep_projection_grid_v3_dev/items.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "qces_public22_identity_residual_blend_v1",
    )
    parser.add_argument("--seed", type=int, default=2264)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_rows(path)
    result = {str(row["event_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate event IDs in {path}")
    return result


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
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _metrics(gain: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    return {
        "events": int(len(gain)),
        "gain_over_crop_db_↑": _summary(gain.tolist()),
        "positive_rate_↑": float(np.mean(gain > 0.0)),
        "nonnegative_rate_↑": float(np.mean(gain >= 0.0)),
        "harmful_below_minus_1db_rate_↓": float(np.mean(gain < -1.0)),
        "edit_coverage_↑": float(np.mean(alpha > 0.0)),
        "mean_alpha": float(np.mean(alpha)),
        "alpha_counts": {
            f"{value:.2f}": int(np.sum(np.isclose(alpha, value)))
            for value in (0.0, *ALPHAS)
        },
    }


def _basic_feature_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names = sorted(rows[0]["features"])
    return [
        name for name in names
        if name in {
            "condition_score", "condition_score_squared", "condition_active_fraction",
            "audiosep_projection_scale",
        }
        or name.endswith("_log_energy")
        or name.endswith("_crop_log_energy_ratio")
    ]


def _join(
    feature_path: Path, grid_path: Path, labels: Sequence[str], feature_names: Sequence[str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_rows = _read_rows(feature_path)
    grid = _read_by_id(grid_path)
    label_to_id = {label: index for index, label in enumerate(labels)}
    x = np.zeros((len(feature_rows), len(labels) + len(feature_names)), dtype=np.float32)
    y = np.zeros((len(feature_rows), len(ALPHAS)), dtype=np.float32)
    groups = np.empty(len(feature_rows), dtype=object)
    proxy = np.zeros(len(feature_rows), dtype=bool)
    for index, row in enumerate(feature_rows):
        event_id = str(row["event_id"])
        if event_id not in grid:
            raise ValueError(f"grid is missing {event_id}")
        grid_row = grid[event_id]
        if str(grid_row["label"]) != str(row["label"]):
            raise ValueError(f"label mismatch for {event_id}")
        x[index, label_to_id[str(row["label"])]] = 1.0
        for offset, name in enumerate(feature_names, len(labels)):
            x[index, offset] = float(row["features"].get(name, 0.0))
        crop = float(grid_row["sd_sdr_db"]["crop"])
        y[index] = [
            float(grid_row["sd_sdr_db"][f"projected_blend_{alpha:.2f}"]) - crop
            for alpha in ALPHAS
        ]
        groups[index] = str(row["source_id"])
        proxy[index] = bool(row["is_proxy"])
    if set(grid) != {str(row["event_id"]) for row in feature_rows}:
        raise ValueError("feature/grid event sets differ")
    return feature_rows, np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0), y, groups, proxy


def _new_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=180, learning_rate=0.04, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=3.0, random_state=seed,
    )


def _new_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=140, learning_rate=0.04, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=3.0, random_state=seed,
    )


def _fit(x: np.ndarray, y: np.ndarray, seed: int) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for index in range(len(ALPHAS)):
        gain = _new_regressor(seed + index * 3)
        positive = _new_classifier(seed + index * 3 + 1)
        harmful = _new_classifier(seed + index * 3 + 2)
        gain.fit(x, np.clip(y[:, index], -20.0, 20.0))
        positive.fit(x, y[:, index] > 0.0)
        harmful.fit(x, y[:, index] < -1.0)
        models.append({"gain": gain, "positive": positive, "harmful": harmful})
    return models


def _predict(models: Sequence[Mapping[str, Any]], x: np.ndarray) -> tuple[np.ndarray, ...]:
    gain = np.column_stack([model["gain"].predict(x) for model in models])
    positive = np.column_stack([model["positive"].predict_proba(x)[:, 1] for model in models])
    harmful = np.column_stack([model["harmful"].predict_proba(x)[:, 1] for model in models])
    return gain, positive, harmful


def _policy(
    prediction: tuple[np.ndarray, ...], actual: np.ndarray,
    setting: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    predicted_gain, positive_probability, harmful_probability = prediction
    minimum_gain, minimum_positive, maximum_harmful = setting
    utility = predicted_gain - RISK_COST_DB * harmful_probability
    choice = np.argmax(utility, axis=1)
    rows = np.arange(len(choice))
    edit = (
        (predicted_gain[rows, choice] >= minimum_gain)
        & (positive_probability[rows, choice] >= minimum_positive)
        & (harmful_probability[rows, choice] <= maximum_harmful)
    )
    gain = np.where(edit, actual[rows, choice], 0.0)
    alpha = np.where(edit, np.asarray(ALPHAS)[choice], 0.0)
    return gain, alpha


def _wilson_upper(successes: int, total: int, z: float = 1.96) -> float:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total)
    return (center + radius) / denominator


def _settings() -> list[tuple[float, float, float]]:
    return [
        (gain, positive, harmful)
        for gain in (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        for positive in (0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70)
        for harmful in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0)
    ]


def _lock_policy(
    prediction: tuple[np.ndarray, ...], actual: np.ndarray,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    eligible: list[tuple[tuple[float, ...], tuple[float, float, float], dict[str, Any], float]] = []
    for setting in _settings():
        gain, alpha = _policy(prediction, actual, setting)
        metrics = _metrics(gain, alpha)
        upper = _wilson_upper(int(np.sum(gain < -1.0)), len(gain))
        if float(metrics["gain_over_crop_db_↑"]["mean"]) < 0.0 or upper > 0.15:
            continue
        median = float(metrics["gain_over_crop_db_↑"]["median"])
        positive = float(metrics["positive_rate_↑"])
        mean = float(metrics["gain_over_crop_db_↑"]["mean"])
        harmful = float(metrics["harmful_below_minus_1db_rate_↓"])
        passes = float(median >= 1.0 and positive >= 0.60)
        bottleneck = min(median / 1.0, positive / 0.60)
        eligible.append(((passes, bottleneck, mean, -harmful), setting, metrics, upper))
    if not eligible:
        raise RuntimeError("no OOF residual policy passes the Wilson safety constraint")
    _objective, setting, metrics, upper = max(eligible, key=lambda row: row[0])
    return setting, {"metrics": metrics, "harmful_wilson_95_upper": upper}


def _candidate_audit(y: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for index, alpha in enumerate(ALPHAS):
        report[f"fixed_alpha_{alpha:.2f}"] = _metrics(y[:, index], np.full(len(y), alpha))
    best_index = np.argmax(np.column_stack([np.zeros(len(y)), y]), axis=1)
    oracle_gain = np.column_stack([np.zeros(len(y)), y])[np.arange(len(y)), best_index]
    oracle_alpha = np.asarray((0.0, *ALPHAS))[best_index]
    report["oracle_alpha_grid"] = _metrics(oracle_gain, oracle_alpha)
    return report


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_features": args.train_features.resolve(),
        "dev_features": args.dev_features.resolve(),
        "train_grid": args.train_grid.resolve(),
        "dev_grid": args.dev_grid.resolve(),
    }
    train_rows = _read_rows(paths["train_features"])
    labels = sorted({str(row["label"]) for row in train_rows})
    feature_names = _basic_feature_names(train_rows)
    train_rows, train_x, train_y, groups, train_proxy = _join(
        paths["train_features"], paths["train_grid"], labels, feature_names,
    )
    dev_rows, dev_x, dev_y, _dev_groups, dev_proxy = _join(
        paths["dev_features"], paths["dev_grid"], labels, feature_names,
    )
    exact_train = ~train_proxy
    exact_dev = ~dev_proxy
    x, y, grouped = train_x[exact_train], train_y[exact_train], groups[exact_train]
    oof = tuple(np.zeros_like(y) for _ in range(3))
    for fold, (fit_index, held_index) in enumerate(GroupKFold(5).split(x, groups=grouped)):
        models = _fit(x[fit_index], y[fit_index], args.seed + fold * 100)
        prediction = _predict(models, x[held_index])
        for destination, values in zip(oof, prediction):
            destination[held_index] = values
    setting, oof_audit = _lock_policy(oof, y)
    models = _fit(x, y, args.seed + 1000)
    dev_prediction = _predict(models, dev_x[exact_dev])
    dev_gain, dev_alpha = _policy(dev_prediction, dev_y[exact_dev], setting)
    dev_metrics = _metrics(dev_gain, dev_alpha)
    decisions: list[dict[str, Any]] = []
    for local_index, row_index in enumerate(np.flatnonzero(exact_dev)):
        decisions.append({
            "event_id": str(dev_rows[row_index]["event_id"]),
            "source_id": str(dev_rows[row_index]["source_id"]),
            "label": str(dev_rows[row_index]["label"]),
            "alpha": float(dev_alpha[local_index]),
            "offline_gain_over_crop_db": float(dev_gain[local_index]),
        })

    gate = {
        "median_gain_db_min": 1.0,
        "mean_gain_db_min": 0.0,
        "positive_rate_min": 0.60,
        "harmful_below_minus_1db_rate_max": 0.15,
    }
    gain_summary = dev_metrics["gain_over_crop_db_↑"]
    accepted = (
        float(gain_summary["median"]) >= gate["median_gain_db_min"]
        and float(gain_summary["mean"]) >= gate["mean_gain_db_min"]
        and float(dev_metrics["positive_rate_↑"]) >= gate["positive_rate_min"]
        and float(dev_metrics["harmful_below_minus_1db_rate_↓"]) <= gate["harmful_below_minus_1db_rate_max"]
    )
    model_path = output / "residual_blend.joblib"
    joblib.dump({
        "format": FORMAT,
        "labels": labels,
        "feature_names": feature_names,
        "alphas": ALPHAS,
        "risk_cost_db": RISK_COST_DB,
        "policy": {
            "minimum_predicted_gain_db": setting[0],
            "minimum_positive_probability": setting[1],
            "maximum_harmful_probability": setting[2],
        },
        "models": models,
    }, model_path)
    decisions_path = output / "dev_decisions.jsonl"
    _write_jsonl(decisions_path, decisions)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "equation": "evidence = crop + alpha * (AudioSep_projected - crop)",
        "alphas": ALPHAS,
        "features": {"label_one_hot": len(labels), "numeric": feature_names},
        "protocol": {
            "target_used_at_inference": False,
            "condition_iou_used_at_inference": False,
            "grouped_oof_by": "source_id",
            "oof_harmful_constraint": "95% Wilson upper bound <= 0.15",
            "policy_selected_on_dev": False,
            "proxy_semantics": "alpha=0 crop fallback",
        },
        "train_oof": oof_audit,
        "locked_policy": {
            "minimum_predicted_gain_db": setting[0],
            "minimum_positive_probability": setting[1],
            "maximum_harmful_probability": setting[2],
            "risk_cost_db": RISK_COST_DB,
        },
        "dev_exact": dev_metrics,
        "dev_candidate_audit": _candidate_audit(dev_y[exact_dev]),
        "dev_acceptance_gate": {**gate, "passed": accepted},
        "decision": "candidate_for_public22_beta" if accepted else "keep_crop_fallback",
        "model": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "dev_decisions": str(decisions_path.resolve()),
        "dev_decisions_sha256": _sha256(decisions_path),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "train_oof": oof_audit,
        "locked_policy": receipt["locked_policy"],
        "dev_exact": dev_metrics,
        "gate": receipt["dev_acceptance_gate"],
        "decision": receipt["decision"],
        "receipt": str((output / "receipt.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
