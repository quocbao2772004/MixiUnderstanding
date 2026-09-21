#!/usr/bin/env python3
"""Train a decoupled no-evidence head and apply it to activity-refined evidence.

The activity-IoU policy intentionally keeps thresholds low to maximise evidence
recall/IoU, which makes no-evidence poor.  This script keeps that span selector
unchanged and learns a separate record-level abstention model:

```
pair/event proposal features -> P(no_evidence)
activity-refined selected_events stay unchanged unless the noev head abstains
```

No validation/test annotations are used to select the no-evidence threshold.
The model is trained on train-fit families and thresholded on train-calibration
families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.scripts.calibrate_qces_v6_span_iou_refiner import (
    event_to_span,
    gold_spans,
    temporal_iou,
)
from mixi_understanding.scripts.qces_v6_firstpair_candidate_utils import (
    split_candidates_firstpair,
)
from mixi_understanding.scripts.train_qces_v6_pair_reranker import (
    Candidate,
    family_calibration_mask,
    prepare_split,
    scores_for,
)


FORMAT_VERSION = "qces_v6_firstpair_noev_head_on_activity_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path, required=True)
    parser.add_argument(
        "--eval",
        action="append",
        nargs=4,
        metavar=("NAME", "MANIFEST", "CACHE", "ACTIVITY_JSONL"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calib-family-frac", type=float, default=0.2)
    parser.add_argument(
        "--min-noev",
        type=float,
        action="append",
        default=[0.60, 0.70, 0.75],
        help="Minimum no-evidence recall target on train-calibration families.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def records_from_manifest(path: Path) -> list[QCESV5Record]:
    dataset = QCESManifestDataset(path, crop_samples=None)
    return [record for record in dataset.records if isinstance(record, QCESV5Record)]


def grouped_scores(
    candidates: Sequence[Candidate], scores: np.ndarray
) -> dict[str, list[tuple[float, Candidate]]]:
    grouped: dict[str, list[tuple[float, Candidate]]] = {}
    for score, candidate in zip(scores, candidates):
        grouped.setdefault(candidate.sample_id, []).append((float(score), candidate))
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0], reverse=True)
    return grouped


def union_length(events: Sequence[Any]) -> float:
    spans = sorted(
        (float(event.onset_seconds), float(event.offset_seconds))
        for event in events
        if event.offset_seconds > event.onset_seconds
    )
    merged: list[list[float]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return float(sum(end - start for start, end in merged))


def record_features(
    records: Sequence[QCESV5Record],
    candidates: Sequence[Candidate],
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[QCESV5Record], dict[str, list[tuple[float, Candidate]]]]:
    grouped = grouped_scores(candidates, scores)
    features: list[list[float]] = []
    targets: list[int] = []
    kept: list[QCESV5Record] = []
    feature_dim = 3 + 12 + 19 + 11
    for record in records:
        rows = grouped.get(record.sample_id, [])
        rel = [
            float(record.relation == "after"),
            float(record.relation == "before"),
            float(record.relation == "first"),
        ]
        if not rows:
            features.append(rel + [0.0] * (feature_dim - 3))
            targets.append(int(record.no_evidence))
            kept.append(record)
            continue
        score_values = np.asarray([row[0] for row in rows], dtype=np.float32)
        top_score, top_candidate = rows[0]
        second = rows[1][0] if len(rows) > 1 else 0.0
        third = rows[2][0] if len(rows) > 2 else 0.0
        quantiles = np.quantile(score_values, [0.25, 0.50, 0.75]).astype(np.float32)
        score_stats = [
            float(top_score),
            float(second),
            float(third),
            float(top_score - second),
            float(top_score - float(score_values.mean())),
            float(score_values.mean()),
            float(score_values.std()),
            float(score_values.min()),
            float(len(rows)),
            float((score_values >= 0.2).mean()),
            float((score_values >= 0.5).mean()),
            float((score_values >= 0.8).mean()),
        ]
        events = list(top_candidate.events)
        confs = np.asarray([float(event.confidence) for event in events], dtype=np.float32)
        durations = np.asarray(
            [float(event.offset_seconds - event.onset_seconds) for event in events],
            dtype=np.float32,
        )
        if events:
            earliest = min(float(event.onset_seconds) for event in events)
            latest = max(float(event.offset_seconds) for event in events)
            event_stats = [
                float(len(events)),
                float(confs.mean()),
                float(confs.min()),
                float(confs.max()),
                float(durations.mean()),
                float(durations.min()),
                float(durations.max()),
                float(union_length(events)),
                float(earliest / 10.0),
                float(latest / 10.0),
                float((latest - earliest) / 10.0),
            ]
        else:
            event_stats = [0.0] * 11
        features.append(rel + score_stats + list(top_candidate.features) + event_stats)
        targets.append(int(record.no_evidence))
        kept.append(record)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.int8),
        kept,
        grouped,
    )


def build_models() -> dict[str, Any]:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.06,
            l2_regularization=0.03,
            max_leaf_nodes=15,
            random_state=31,
        ),
        "rf": RandomForestClassifier(
            n_estimators=260,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=32,
        ),
        "extra": ExtraTreesClassifier(
            n_estimators=320,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=33,
        ),
    }


def evaluate_activity_override(
    *,
    records: Sequence[QCESV5Record],
    activity_rows: Sequence[Mapping[str, Any]],
    noev_probs: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {record.sample_id: record for record in records}
    answer_total = answer_correct = noev_total = noev_correct = decision_correct = 0
    iou_sum = raw_iou_sum = iou_count = 0
    abstain_answerable = 0
    items: list[dict[str, Any]] = []
    by_relation: dict[str, dict[str, float]] = {}
    for row, prob in zip(activity_rows, noev_probs):
        record = by_id[row["id"]]
        predicted_noev = float(prob) >= threshold
        selected_events = [] if predicted_noev else list(row.get("selected_events") or [])
        selected_raw = [] if predicted_noev else list(row.get("selected_events_raw") or [])
        predicted_answer = None if predicted_noev else row.get("predicted_answer")
        decision_correct += int(predicted_noev == bool(record.no_evidence))
        answer_ok: bool | None
        raw_iou: float | None
        refined_iou: float | None
        if record.no_evidence:
            noev_total += 1
            noev_correct += int(predicted_noev)
            answer_ok = None
            raw_iou = refined_iou = None
        else:
            answer_total += 1
            answer_ok = (not predicted_noev) and predicted_answer == record.answer
            answer_correct += int(answer_ok)
            abstain_answerable += int(predicted_noev)
            target = gold_spans(record)
            raw_iou = (
                0.0
                if predicted_noev
                else temporal_iou(
                    [
                        (float(event["onset_seconds"]), float(event["offset_seconds"]))
                        for event in selected_raw
                    ],
                    target,
                )
            )
            refined_iou = (
                0.0
                if predicted_noev
                else temporal_iou(
                    [
                        (float(event["onset_seconds"]), float(event["offset_seconds"]))
                        for event in selected_events
                    ],
                    target,
                )
            )
            raw_iou_sum += raw_iou
            iou_sum += refined_iou
            iou_count += 1
            rel = record.relation
            bucket = by_relation.setdefault(
                rel, {"answer": 0.0, "count": 0.0, "iou": 0.0}
            )
            bucket["answer"] += float(answer_ok)
            bucket["count"] += 1.0
            bucket["iou"] += refined_iou
        out = dict(row)
        out.update(
            {
                "base_predicted_no_evidence": row.get("predicted_no_evidence"),
                "base_predicted_answer": row.get("predicted_answer"),
                "noev_head_probability": float(prob),
                "noev_head_threshold": float(threshold),
                "predicted_no_evidence": bool(predicted_noev),
                "predicted_answer": predicted_answer,
                "answer_correct": answer_ok,
                "span_iou_raw_↑": raw_iou,
                "span_iou_↑": refined_iou,
                "span_iou_delta_↑": (
                    None if raw_iou is None or refined_iou is None else refined_iou - raw_iou
                ),
                "selected_events_raw": selected_raw,
                "selected_events": selected_events,
            }
        )
        items.append(out)
    summary = {
        "record_count": len(activity_rows),
        "answerable_count": answer_total,
        "no_evidence_count": noev_total,
        "answer_accuracy_↑": answer_correct / answer_total if answer_total else None,
        "no_evidence_accuracy_↑": noev_correct / noev_total if noev_total else None,
        "decision_accuracy_↑": decision_correct / len(activity_rows) if activity_rows else None,
        "abstain_rate_on_answerable_↓": abstain_answerable / answer_total if answer_total else None,
        "span_iou_raw_mean_↑": raw_iou_sum / iou_count if iou_count else None,
        "span_iou_refined_mean_↑": iou_sum / iou_count if iou_count else None,
        "span_iou_refined_by_relation_↑": {
            rel: values["iou"] / values["count"]
            for rel, values in sorted(by_relation.items())
            if values["count"]
        },
        "answer_accuracy_by_relation_↑": {
            rel: values["answer"] / values["count"]
            for rel, values in sorted(by_relation.items())
            if values["count"]
        },
    }
    return summary, items


def choose_policy(
    *,
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    calib_features: np.ndarray,
    calib_records: Sequence[QCESV5Record],
    calib_activity_rows: Sequence[Mapping[str, Any]],
    min_noev: float,
) -> tuple[str, float, Any, dict[str, Any]]:
    best: tuple[tuple[float, float, float, float], str, float, Any, dict[str, Any]] | None = None
    best_any: tuple[tuple[float, float, float, float], str, float, Any, dict[str, Any]] | None = None
    for name, model in build_models().items():
        model.fit(fit_features, fit_targets)
        probs = np.asarray(model.predict_proba(calib_features)[:, 1], dtype=np.float32)
        thresholds = np.unique(
            np.r_[
                np.linspace(0.0, 1.0, 101),
                np.quantile(probs, np.linspace(0.0, 1.0, 101)),
            ]
        )
        for threshold in thresholds:
            summary, _ = evaluate_activity_override(
                records=calib_records,
                activity_rows=calib_activity_rows,
                noev_probs=probs,
                threshold=float(threshold),
            )
            key = (
                float(summary["span_iou_refined_mean_↑"]),
                float(summary["answer_accuracy_↑"]),
                float(summary["no_evidence_accuracy_↑"]),
                -float(summary["abstain_rate_on_answerable_↓"]),
            )
            if best_any is None or key > best_any[0]:
                best_any = (key, name, float(threshold), model, summary)
            if float(summary["no_evidence_accuracy_↑"]) >= min_noev:
                if best is None or key > best[0]:
                    best = (key, name, float(threshold), model, summary)
    chosen = best or best_any
    assert chosen is not None
    return chosen[1], chosen[2], chosen[3], chosen[4]


def prepare_features_for_manifest(
    *,
    manifest: Path,
    cache: Path,
    dataset_config: Path,
    proposal_head: Path,
    relation_thresholds: Mapping[str, float],
    pair_model: Any,
    max_neighbors: int,
    device: torch.device,
) -> tuple[list[QCESV5Record], np.ndarray, np.ndarray, dict[str, list[tuple[float, Candidate]]]]:
    prepared = prepare_split(
        manifest=manifest.resolve(),
        cache_path=cache.resolve(),
        dataset_config=dataset_config.resolve(),
        proposal_head=proposal_head.resolve(),
        thresholds=relation_thresholds,
        device=device,
    )
    features, _targets, candidates, _records_by_id = split_candidates_firstpair(
        prepared,
        max_neighbors=max_neighbors,
    )
    scores = scores_for(pair_model, features)
    rec_features, targets, records, grouped = record_features(
        prepared[0],
        candidates,
        scores,
    )
    return records, rec_features, targets, grouped


def activity_rows_for_records(
    activity_jsonl: Path,
    records: Sequence[QCESV5Record],
) -> list[dict[str, Any]]:
    rows_by_id = {row["id"]: row for row in read_jsonl(activity_jsonl)}
    missing = [record.sample_id for record in records if record.sample_id not in rows_by_id]
    if missing:
        raise SystemExit(f"activity jsonl misses {len(missing)} records, e.g. {missing[:3]}")
    return [rows_by_id[record.sample_id] for record in records]


def markdown(report: Mapping[str, Any]) -> str:
    def f(value: Any) -> str:
        return "..." if value is None else f"{float(value):.3f}"

    lines = [
        "# QCES-v6 decoupled no-evidence head on activity evidence",
        "",
        "| Split | Policy | model | thr | answer ↑ | no-evid ↑ | decision ↑ | IoU raw ↑ | IoU refined ↑ | abstain answerable ↓ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, policies in report["eval_results"].items():
        for policy_name, row in policies.items():
            summary = row["summary"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        policy_name,
                        row["model_name"],
                        f(row["threshold"]),
                        f(summary["answer_accuracy_↑"]),
                        f(summary["no_evidence_accuracy_↑"]),
                        f(summary["decision_accuracy_↑"]),
                        f(summary["span_iou_raw_mean_↑"]),
                        f(summary["span_iou_refined_mean_↑"]),
                        f(summary["abstain_rate_on_answerable_↓"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    payload = joblib.load(args.model.resolve())
    pair_model = payload["model"]
    relation_thresholds = payload["relation_thresholds"]
    max_neighbors = int(payload["max_neighbors"])

    train_prepared = prepare_split(
        manifest=args.train_manifest.resolve(),
        cache_path=args.train_cache.resolve(),
        dataset_config=args.dataset_config.resolve(),
        proposal_head=args.proposal_head.resolve(),
        thresholds=relation_thresholds,
        device=device,
    )
    train_features_pair, _targets, train_candidates, train_records_by_id = (
        split_candidates_firstpair(train_prepared, max_neighbors=max_neighbors)
    )
    train_scores = scores_for(pair_model, train_features_pair)
    train_rec_features, train_noev_targets, train_records, _grouped = record_features(
        train_prepared[0],
        train_candidates,
        train_scores,
    )

    calib_mask_candidates = family_calibration_mask(
        train_candidates,
        train_records_by_id,
        frac=args.calib_family_frac,
    )
    calib_ids = {
        candidate.sample_id
        for candidate, keep in zip(train_candidates, calib_mask_candidates)
        if keep
    }
    calib_id_set = set(calib_ids)
    calib_mask_records = np.asarray(
        [record.sample_id in calib_id_set for record in train_records],
        dtype=bool,
    )
    fit_mask_records = ~calib_mask_records
    if not bool(fit_mask_records.any()) or not bool(calib_mask_records.any()):
        raise SystemExit("empty fit/calibration split")

    # Train-calibration has no activity-refined output, so use the top raw
    # candidate events as a calibration proxy.  The threshold is still selected
    # only on train families and later applied to real activity-refined eval
    # evidence.
    calib_activity_proxy: list[dict[str, Any]] = []
    grouped = grouped_scores(train_candidates, train_scores)
    for record in np.asarray(train_records, dtype=object)[calib_mask_records]:
        rows = grouped.get(record.sample_id, [])
        if rows:
            score, candidate = rows[0]
            selected = [
                {
                    "label": event.label,
                    "onset_seconds": event.onset_seconds,
                    "offset_seconds": event.offset_seconds,
                    "confidence": event.confidence,
                }
                for event in candidate.events
            ]
            pred_answer = candidate.answer_label
        else:
            score = None
            selected = []
            pred_answer = None
        calib_activity_proxy.append(
            {
                "id": record.sample_id,
                "question": record.question,
                "relation": record.relation,
                "answer": record.answer,
                "no_evidence": bool(record.no_evidence),
                "predicted_no_evidence": False,
                "predicted_answer": pred_answer,
                "score": score,
                "selected_events_raw": selected,
                "selected_events": selected,
            }
        )

    policies: dict[str, Any] = {}
    for min_noev in sorted(set(float(value) for value in args.min_noev)):
        name, threshold, model, calibration = choose_policy(
            fit_features=train_rec_features[fit_mask_records],
            fit_targets=train_noev_targets[fit_mask_records],
            calib_features=train_rec_features[calib_mask_records],
            calib_records=list(np.asarray(train_records, dtype=object)[calib_mask_records]),
            calib_activity_rows=calib_activity_proxy,
            min_noev=min_noev,
        )
        final_model = build_models()[name]
        final_model.fit(train_rec_features, train_noev_targets)
        policies[f"noev_head_min{min_noev:.2f}"] = {
            "model_name": name,
            "threshold": threshold,
            "model": final_model,
            "calibration_summary": calibration,
            "min_noev": min_noev,
        }

    eval_results: dict[str, Any] = {}
    for split_name, manifest_text, cache_text, activity_text in args.eval:
        records, rec_features, _targets_eval, _grouped_eval = prepare_features_for_manifest(
            manifest=Path(manifest_text),
            cache=Path(cache_text),
            dataset_config=args.dataset_config,
            proposal_head=args.proposal_head,
            relation_thresholds=relation_thresholds,
            pair_model=pair_model,
            max_neighbors=max_neighbors,
            device=device,
        )
        activity_rows = activity_rows_for_records(Path(activity_text), records)
        eval_results[split_name] = {}
        for policy_name, policy in policies.items():
            probs = np.asarray(
                policy["model"].predict_proba(rec_features)[:, 1],
                dtype=np.float32,
            )
            summary, items = evaluate_activity_override(
                records=records,
                activity_rows=activity_rows,
                noev_probs=probs,
                threshold=float(policy["threshold"]),
            )
            eval_results[split_name][policy_name] = {
                "model_name": policy["model_name"],
                "threshold": float(policy["threshold"]),
                "min_noev": float(policy["min_noev"]),
                "summary": summary,
            }
            write_jsonl(out / f"{split_name}__{policy_name}.jsonl", items)

    slim_policies = {
        name: {
            key: value
            for key, value in policy.items()
            if key not in {"model"}
        }
        for name, policy in policies.items()
    }
    report = {
        "format": FORMAT_VERSION,
        "device": str(device),
        "pair_model": str(args.model.resolve()),
        "pair_model_sha256": sha256_file(args.model.resolve()),
        "proposal_head": str(args.proposal_head.resolve()),
        "proposal_head_sha256": sha256_file(args.proposal_head.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "relation_thresholds": relation_thresholds,
        "max_neighbors": max_neighbors,
        "calib_family_frac": args.calib_family_frac,
        "policies": slim_policies,
        "eval_results": eval_results,
    }
    (out / "noev_head_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "noev_head_results.md").write_text(markdown(report), encoding="utf-8")
    joblib.dump(
        {
            "format": FORMAT_VERSION,
            "pair_model_payload": payload,
            "proposal_head": str(args.proposal_head.resolve()),
            "policies": policies,
        },
        out / "noev_head_policy.joblib",
    )
    print(markdown(report), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
