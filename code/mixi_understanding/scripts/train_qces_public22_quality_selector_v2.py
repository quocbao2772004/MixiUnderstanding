#!/usr/bin/env python3
"""Train a source-grouped, risk-aware selector from v2 deployable features.

For each non-crop candidate the selector learns three quantities: clipped
SD-SDR gain, probability of a positive gain, and probability of harmful loss
below -1 dB.  The inference policy is selected exclusively from grouped
out-of-fold train predictions.  Development data is opened once afterward.
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


FORMAT = "qces_public22_quality_selector_v2"
CANDIDATES = ("crop", "custom_span_postcrop", "audiosep_projected")
REFINEMENTS = CANDIDATES[1:]
RISK_COST_DB = 5.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3] / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-items", type=Path,
        default=root / "qces_public22_selector_features_v2_train/items.jsonl",
    )
    parser.add_argument(
        "--dev-items", type=Path,
        default=root / "qces_public22_selector_features_v2_dev/items.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "qces_public22_quality_selector_v2",
    )
    parser.add_argument("--seed", type=int, default=2262)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def _feature_groups(feature_names: Sequence[str]) -> dict[str, list[str]]:
    basic = [
        name for name in feature_names
        if name in {
            "condition_score", "condition_score_squared", "condition_active_fraction",
            "audiosep_projection_scale",
        }
        or name.endswith("_log_energy")
        or name.endswith("_crop_log_energy_ratio")
    ]
    waveform = [name for name in feature_names if not name.startswith("beats_")]
    return {"basic": sorted(basic), "waveform": sorted(waveform), "full": list(feature_names)}


def _matrix(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[str], numeric_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    x = np.zeros((len(rows), len(labels) + len(numeric_names)), dtype=np.float32)
    y = np.zeros((len(rows), len(REFINEMENTS)), dtype=np.float32)
    groups = np.empty(len(rows), dtype=object)
    proxy = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        label = str(row["label"])
        if label not in label_to_id:
            raise ValueError(f"unseen dev label: {label}")
        x[index, label_to_id[label]] = 1.0
        features = row["features"]
        for offset, name in enumerate(numeric_names, len(labels)):
            x[index, offset] = float(features.get(name, 0.0))
        quality = row["offline_quality_labels"]["gain_over_crop_db"]
        y[index] = [float(quality[name]) for name in REFINEMENTS]
        groups[index] = str(row["source_id"])
        proxy[index] = bool(row["is_proxy"])
    x = np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
    return x, y, groups, proxy


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


def _fit_models(x: np.ndarray, y: np.ndarray, seed: int) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    clipped = np.clip(y, -20.0, 20.0)
    for candidate_index in range(len(REFINEMENTS)):
        gain = _new_regressor(seed + candidate_index * 3)
        positive = _new_classifier(seed + candidate_index * 3 + 1)
        harmful = _new_classifier(seed + candidate_index * 3 + 2)
        gain.fit(x, clipped[:, candidate_index])
        positive.fit(x, y[:, candidate_index] > 0.0)
        harmful.fit(x, y[:, candidate_index] < -1.0)
        models.append({"gain": gain, "positive": positive, "harmful": harmful})
    return models


def _predict(models: Sequence[Mapping[str, Any]], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gain = np.column_stack([model["gain"].predict(x) for model in models])
    positive = np.column_stack([model["positive"].predict_proba(x)[:, 1] for model in models])
    harmful = np.column_stack([model["harmful"].predict_proba(x)[:, 1] for model in models])
    return gain, positive, harmful


def _policy(
    prediction: tuple[np.ndarray, np.ndarray, np.ndarray],
    actual: np.ndarray,
    setting: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    predicted_gain, positive_probability, harmful_probability = prediction
    minimum_gain, minimum_positive, maximum_harmful = setting
    risk_adjusted = predicted_gain - RISK_COST_DB * harmful_probability
    selected = np.argmax(risk_adjusted, axis=1)
    index = np.arange(len(selected))
    use = (
        (predicted_gain[index, selected] >= minimum_gain)
        & (positive_probability[index, selected] >= minimum_positive)
        & (harmful_probability[index, selected] <= maximum_harmful)
    )
    gains = np.where(use, actual[index, selected], 0.0)
    choices = np.where(use, selected + 1, 0)
    return gains, choices


def _metrics(gains: np.ndarray, choices: np.ndarray) -> dict[str, Any]:
    return {
        "events": int(len(gains)),
        "gain_over_crop_db_↑": _summary(gains.tolist()),
        "positive_rate_↑": float(np.mean(gains > 0.0)),
        "nonnegative_rate_↑": float(np.mean(gains >= 0.0)),
        "harmful_below_minus_1db_rate_↓": float(np.mean(gains < -1.0)),
        "refinement_coverage_↑": float(np.mean(choices != 0)),
        "selection_counts": {
            CANDIDATES[index]: int(np.sum(choices == index))
            for index in range(len(CANDIDATES))
        },
    }


def _setting_grid() -> list[tuple[float, float, float]]:
    return [
        (gain, positive, harmful)
        for gain in (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
        for positive in (0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70)
        for harmful in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0)
    ]


def _select_setting(
    prediction: tuple[np.ndarray, np.ndarray, np.ndarray], actual: np.ndarray,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    table: dict[str, Any] = {}
    eligible: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for setting in _setting_grid():
        gains, choices = _policy(prediction, actual, setting)
        metrics = _metrics(gains, choices)
        key = f"gain={setting[0]:.2f}|positive={setting[1]:.2f}|harmful={setting[2]:.2f}"
        table[key] = metrics
        if (
            float(metrics["gain_over_crop_db_↑"]["mean"]) >= 0.0
            and float(metrics["harmful_below_minus_1db_rate_↓"]) <= 0.15
        ):
            eligible.append((setting, metrics))
    if not eligible:
        raise RuntimeError("no source-grouped OOF policy satisfies mean and harmful-rate constraints")

    def objective(item: tuple[tuple[float, float, float], Mapping[str, Any]]) -> tuple[float, ...]:
        _setting, metrics = item
        gain = metrics["gain_over_crop_db_↑"]
        median = float(gain["median"])
        positive = float(metrics["positive_rate_↑"])
        mean = float(gain["mean"])
        harmful = float(metrics["harmful_below_minus_1db_rate_↓"])
        passes_shape = float(median >= 1.0 and positive >= 0.60)
        normalized_bottleneck = min(median / 1.0, positive / 0.60)
        return passes_shape, normalized_bottleneck, mean, -harmful

    selected, metrics = max(eligible, key=objective)
    return selected, {"selected": metrics, "grid": table}


def _run_variant(
    train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str], numeric_names: Sequence[str], seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    train_x, train_y, groups, train_proxy = _matrix(train_rows, labels, numeric_names)
    dev_x, dev_y, _dev_groups, dev_proxy = _matrix(dev_rows, labels, numeric_names)
    exact_train = ~train_proxy
    exact_dev = ~dev_proxy
    x = train_x[exact_train]
    y = train_y[exact_train]
    grouped = groups[exact_train]
    oof_gain = np.zeros_like(y)
    oof_positive = np.zeros_like(y)
    oof_harmful = np.zeros_like(y)
    splitter = GroupKFold(n_splits=5)
    for fold, (fit_index, held_index) in enumerate(splitter.split(x, groups=grouped)):
        models = _fit_models(x[fit_index], y[fit_index], seed + fold * 100)
        prediction = _predict(models, x[held_index])
        oof_gain[held_index], oof_positive[held_index], oof_harmful[held_index] = prediction
    setting, setting_audit = _select_setting((oof_gain, oof_positive, oof_harmful), y)
    models = _fit_models(x, y, seed + 1000)
    dev_prediction = _predict(models, dev_x[exact_dev])
    dev_gain, dev_choice = _policy(dev_prediction, dev_y[exact_dev], setting)
    all_gain = np.zeros(len(dev_rows), dtype=np.float64)
    all_choice = np.zeros(len(dev_rows), dtype=np.int64)
    all_gain[exact_dev] = dev_gain
    all_choice[exact_dev] = dev_choice
    diagnostic_rows: list[dict[str, Any]] = []
    exact_indices = np.flatnonzero(exact_dev)
    for local_index, row_index in enumerate(exact_indices):
        choice = int(dev_choice[local_index])
        diagnostic_rows.append({
            "event_id": str(dev_rows[row_index]["event_id"]),
            "source_id": str(dev_rows[row_index]["source_id"]),
            "label": str(dev_rows[row_index]["label"]),
            "selected_candidate": CANDIDATES[choice],
            "selected_gain_over_crop_db": float(dev_gain[local_index]),
            "predicted": {
                candidate: {
                    "gain_db": float(dev_prediction[0][local_index, candidate_index]),
                    "positive_probability": float(dev_prediction[1][local_index, candidate_index]),
                    "harmful_probability": float(dev_prediction[2][local_index, candidate_index]),
                }
                for candidate_index, candidate in enumerate(REFINEMENTS)
            },
            "offline_actual_gain_db": {
                candidate: float(dev_y[row_index, candidate_index])
                for candidate_index, candidate in enumerate(REFINEMENTS)
            },
        })
    report = {
        "numeric_feature_count": len(numeric_names),
        "total_feature_count": len(labels) + len(numeric_names),
        "selected_policy": {
            "minimum_predicted_gain_db": setting[0],
            "minimum_positive_probability": setting[1],
            "maximum_harmful_probability": setting[2],
            "risk_cost_db": RISK_COST_DB,
        },
        "train_oof": setting_audit["selected"],
        "dev_exact": _metrics(dev_gain, dev_choice),
        "dev_all_with_proxy_crop_fallback": _metrics(all_gain, all_choice),
    }
    bundle = {
        "models": models,
        "numeric_feature_names": list(numeric_names),
        "labels": list(labels),
        "policy": report["selected_policy"],
    }
    return report, bundle, diagnostic_rows


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_path = args.train_items.resolve()
    dev_path = args.dev_items.resolve()
    train_rows = _read_jsonl(train_path)
    dev_rows = _read_jsonl(dev_path)
    labels = sorted({str(row["label"]) for row in train_rows})
    train_feature_names = sorted(train_rows[0]["features"])
    dev_feature_names = sorted(dev_rows[0]["features"])
    if train_feature_names != dev_feature_names:
        raise ValueError("train/dev deployable feature schema mismatch")
    variants = _feature_groups(train_feature_names)
    reports: dict[str, Any] = {}
    full_bundle: dict[str, Any] | None = None
    full_diagnostics: list[dict[str, Any]] = []
    for index, (name, numeric_names) in enumerate(variants.items()):
        report, bundle, diagnostics = _run_variant(
            train_rows, dev_rows, labels, numeric_names, args.seed + index * 10_000,
        )
        reports[name] = report
        if name == "full":
            full_bundle = bundle
            full_diagnostics = diagnostics
        print(json.dumps({"variant": name, "dev_exact": report["dev_exact"]}), flush=True)
    assert full_bundle is not None

    exact = reports["full"]["dev_exact"]
    gain = exact["gain_over_crop_db_↑"]
    gate = {
        "median_gain_db_min": 1.0,
        "mean_gain_db_min": 0.0,
        "positive_rate_min": 0.60,
        "harmful_below_minus_1db_rate_max": 0.15,
    }
    accepted = (
        float(gain["median"]) >= gate["median_gain_db_min"]
        and float(gain["mean"]) >= gate["mean_gain_db_min"]
        and float(exact["positive_rate_↑"]) >= gate["positive_rate_min"]
        and float(exact["harmful_below_minus_1db_rate_↓"]) <= gate["harmful_below_minus_1db_rate_max"]
    )
    bundle = {
        "format": FORMAT,
        "candidates": CANDIDATES,
        "risk_cost_db": RISK_COST_DB,
        **full_bundle,
    }
    model_path = output / "selector.joblib"
    joblib.dump(bundle, model_path)
    diagnostics_path = output / "dev_decisions.jsonl"
    _write_jsonl(diagnostics_path, full_diagnostics)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "protocol": {
            "target_used_as_inference_feature": False,
            "condition_iou_used_as_inference_feature": False,
            "proxy_semantics": "forced crop fallback",
            "five_fold_oof_grouping": "source_id",
            "policy_selected_on": "train grouped OOF only",
            "dev_opened_after_policy_lock": True,
            "candidate_gain_clip_db": [-20.0, 20.0],
        },
        "feature_ablation": reports,
        "dev_acceptance_gate": {**gate, "passed": accepted},
        "decision": "candidate_for_public22_beta" if accepted else "keep_crop_fallback",
        "selector": str(model_path.resolve()),
        "selector_sha256": _sha256(model_path),
        "dev_decisions": str(diagnostics_path.resolve()),
        "dev_decisions_sha256": _sha256(diagnostics_path),
        "inputs": {
            "train": {"path": str(train_path), "sha256": _sha256(train_path)},
            "dev": {"path": str(dev_path), "sha256": _sha256(dev_path)},
        },
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "full_dev_exact": exact,
        "gate": receipt["dev_acceptance_gate"],
        "decision": receipt["decision"],
        "receipt": str((output / "receipt.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
