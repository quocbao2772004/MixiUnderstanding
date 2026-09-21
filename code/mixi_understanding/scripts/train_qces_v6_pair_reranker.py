#!/usr/bin/env python3
"""Train a question-conditioned pair reranker for QCES-v6 proposals.

This script is intentionally separate from Claude's original QCES-v6 pipeline.
It keeps the frozen proposal head and symbolic question parser, but replaces the
planner's hard "nearest neighbour by onset" choice with a small supervised
reranker over candidate event proposals.

The model reads only deployable proposal features:

* relation type recovered from the question;
* anchor ordinal recovered from the question;
* proposal onset/offset/confidence from the frozen separator-as-detector cache.

It does not read validation annotations at inference time.  Train annotations
are used only to label candidate pairs for supervised learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import joblib
import numpy as np
import torch
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.qces.event_proposals import (
    EventProposal,
    ProposalHead,
    decode_proposals,
)
from mixi_understanding.qces.question_parsing import ParsedQuestion, parse_question
from mixi_understanding.qces.relational_planner import EvidencePlan, plan_from_proposals
from mixi_understanding.scripts.evaluate_qces_v6_pipeline import (
    FrameGrid,
    interval_iou,
    load_taxonomy,
    scene_activity,
)

FORMAT_VERSION = "qces_v6_pair_reranker_v1"
FEATURE_NAMES = (
    "rel_after",
    "rel_before",
    "rel_first",
    "anchor_or_candidate_conf",
    "neighbor_conf",
    "confidence_product",
    "anchor_or_candidate_onset_norm",
    "neighbor_onset_norm",
    "anchor_or_candidate_duration",
    "neighbor_duration",
    "anchor_or_candidate_occurrence_rank",
    "anchor_rank_abs_error",
    "anchor_rank_exact",
    "num_anchor_or_label_proposals",
    "num_total_proposals",
    "neighbor_distance_rank",
    "anchor_neighbor_distance",
    "log1p_anchor_neighbor_distance",
    "neighbor_same_as_anchor",
)
DEFAULT_RELATION_THRESHOLDS = {"after": 0.12, "before": 0.15, "first": 0.03}


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    answer_label: str
    events: tuple[EventProposal, ...]
    features: tuple[float, ...]
    target: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calib-family-frac", type=float, default=0.2)
    parser.add_argument("--relation-threshold", action="append", default=[])
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["logreg", "hgb", "extra", "rf"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relation_thresholds(values: Sequence[str]) -> dict[str, float]:
    result = dict(DEFAULT_RELATION_THRESHOLDS)
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --relation-threshold {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if key not in result:
            raise SystemExit(f"unknown relation threshold: {key}")
        result[key] = float(raw)
    return result


def record_rows(manifest: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def prepare_split(
    *,
    manifest: Path,
    cache_path: Path,
    dataset_config: Path,
    proposal_head: Path,
    thresholds: Mapping[str, float],
    device: torch.device,
) -> tuple[
    list[QCESV5Record],
    dict[str, ParsedQuestion],
    dict[str, tuple[str, ...]],
    dict[str, dict[tuple[str, tuple[str, ...]], list[EventProposal]]],
]:
    dataset = QCESManifestDataset(manifest, crop_samples=None)
    rows = record_rows(manifest)
    taxonomy = load_taxonomy(dataset_config)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached_scenes = set(cache["scenes"])
    records = [
        record
        for record in dataset.records
        if isinstance(record, QCESV5Record) and record.scene_id in cached_scenes
    ]

    payload = torch.load(proposal_head, map_location="cpu", weights_only=False)
    head = ProposalHead(
        channels=int(payload.get("channels", 96)),
        dropout=float(payload.get("dropout", 0.1)),
    )
    head.load_state_dict(payload["state_dict"])
    head = head.to(device).eval()
    zeroed = tuple(payload.get("zeroed_feature_groups", ()))
    onset_split = bool(payload.get("onset_split", True))
    activity_by_scene = scene_activity(cache, head, device, zeroed)
    grid = FrameGrid(sample_rate=32_000)

    parsed: dict[str, ParsedQuestion] = {}
    labels_by_id: dict[str, tuple[str, ...]] = {}
    for record in records:
        row = rows[record.sample_id]
        question = parse_question(row["question"], row["answer_options"], taxonomy)
        parsed[record.sample_id] = question
        labels_by_id[record.sample_id] = tuple(question.query_labels)

    decoded_by_relation: dict[
        str, dict[tuple[str, tuple[str, ...]], list[EventProposal]]
    ] = {}
    for relation, threshold in thresholds.items():
        decoded: dict[tuple[str, tuple[str, ...]], list[EventProposal]] = {}
        for record in records:
            labels = labels_by_id[record.sample_id]
            key = (record.scene_id, labels)
            if key in decoded:
                continue
            available = [
                label for label in labels if label in activity_by_scene[record.scene_id]
            ]
            if available:
                activity = torch.stack(
                    [activity_by_scene[record.scene_id][label][0] for label in available]
                )
                onsets = [
                    activity_by_scene[record.scene_id][label][1] for label in available
                ]
                onset_activity = (
                    None
                    if onsets[0] is None or not onset_split
                    else torch.stack(onsets)
                )
                proposals = decode_proposals(
                    available,
                    activity,
                    grid,
                    threshold=threshold,
                    onset_activity=onset_activity,
                )
            else:
                proposals = []
            decoded[key] = proposals
        decoded_by_relation[relation] = decoded
    return records, parsed, labels_by_id, decoded_by_relation


def candidate_features(
    record: QCESV5Record,
    parsed: ParsedQuestion,
    proposals: Sequence[EventProposal],
    *,
    max_neighbors: int,
) -> list[tuple[str, tuple[EventProposal, ...], tuple[float, ...]]]:
    if not parsed.ok or parsed.relation is None:
        return []
    relation = parsed.relation
    rel_after = float(relation == "after")
    rel_before = float(relation == "before")
    rel_first = float(relation == "first")
    total = len(proposals)

    if relation == "first":
        rows: list[tuple[str, tuple[EventProposal, ...], tuple[float, ...]]] = []
        candidates: list[tuple[str, EventProposal, int, int]] = []
        for label in parsed.candidate_labels:
            label_props = sorted(
                [item for item in proposals if item.label == label],
                key=lambda item: (item.onset_seconds, item.offset_seconds),
            )
            for occurrence_rank, proposal in enumerate(label_props[:3], start=1):
                candidates.append((label, proposal, occurrence_rank, len(label_props)))
        onset_rank = {
            id(item[1]): position + 1
            for position, item in enumerate(
                sorted(candidates, key=lambda item: item[1].onset_seconds)
            )
        }
        for label, proposal, occurrence_rank, num_label_props in candidates:
            rows.append(
                (
                    label,
                    (proposal,),
                    (
                        rel_after,
                        rel_before,
                        rel_first,
                        proposal.confidence,
                        0.0,
                        proposal.confidence,
                        proposal.onset_seconds / 10.0,
                        0.0,
                        proposal.duration_seconds,
                        0.0,
                        float(occurrence_rank),
                        0.0,
                        float(occurrence_rank == 1),
                        float(num_label_props),
                        float(total),
                        float(onset_rank[id(proposal)]),
                        0.0,
                        0.0,
                        0.0,
                    ),
                )
            )
        return rows

    if parsed.anchor_label is None or parsed.anchor_ordinal is None:
        return []
    anchors = sorted(
        [item for item in proposals if item.label == parsed.anchor_label],
        key=lambda item: (item.onset_seconds, item.offset_seconds),
    )
    rows = []
    for anchor_rank, anchor in enumerate(anchors, start=1):
        if relation == "after":
            neighbors = [
                item
                for item in proposals
                if item is not anchor and item.onset_seconds > anchor.onset_seconds + 1e-6
            ]
        else:
            neighbors = [
                item
                for item in proposals
                if item is not anchor and item.onset_seconds < anchor.onset_seconds - 1e-6
            ]
        selected: list[EventProposal] = []
        seen: set[int] = set()
        for item in sorted(
            neighbors, key=lambda item: abs(item.onset_seconds - anchor.onset_seconds)
        )[:max_neighbors] + sorted(neighbors, key=lambda item: -item.confidence)[
            :max_neighbors
        ]:
            if id(item) in seen:
                continue
            seen.add(id(item))
            selected.append(item)
        distance_rank = {
            id(item): position + 1
            for position, item in enumerate(
                sorted(
                    selected,
                    key=lambda item: abs(item.onset_seconds - anchor.onset_seconds),
                )
            )
        }
        for neighbor in selected:
            distance = abs(neighbor.onset_seconds - anchor.onset_seconds)
            rows.append(
                (
                    neighbor.label,
                    tuple(sorted((anchor, neighbor), key=lambda item: item.onset_seconds)),
                    (
                        rel_after,
                        rel_before,
                        rel_first,
                        anchor.confidence,
                        neighbor.confidence,
                        anchor.confidence * neighbor.confidence,
                        anchor.onset_seconds / 10.0,
                        neighbor.onset_seconds / 10.0,
                        anchor.duration_seconds,
                        neighbor.duration_seconds,
                        float(anchor_rank),
                        float(abs(anchor_rank - parsed.anchor_ordinal)),
                        float(anchor_rank == parsed.anchor_ordinal),
                        float(len(anchors)),
                        float(total),
                        float(distance_rank[id(neighbor)]),
                        distance,
                        float(np.log1p(distance)),
                        float(neighbor.label == anchor.label),
                    ),
                )
            )
    return rows


def split_candidates(
    prepared: tuple[
        list[QCESV5Record],
        dict[str, ParsedQuestion],
        dict[str, tuple[str, ...]],
        dict[str, dict[tuple[str, tuple[str, ...]], list[EventProposal]]],
    ],
    *,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, list[Candidate], dict[str, QCESV5Record]]:
    records, parsed, labels_by_id, decoded_by_relation = prepared
    features: list[tuple[float, ...]] = []
    targets: list[int] = []
    candidates: list[Candidate] = []
    records_by_id = {record.sample_id: record for record in records}
    for record in records:
        question = parsed[record.sample_id]
        if question.relation not in decoded_by_relation:
            continue
        proposals = decoded_by_relation[question.relation][
            (record.scene_id, labels_by_id[record.sample_id])
        ]
        for answer_label, events, feature_row in candidate_features(
            record, question, proposals, max_neighbors=max_neighbors
        ):
            target = int((not record.no_evidence) and answer_label == record.answer)
            candidate = Candidate(
                sample_id=record.sample_id,
                answer_label=answer_label,
                events=events,
                features=feature_row,
                target=target,
            )
            candidates.append(candidate)
            features.append(feature_row)
            targets.append(target)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.int8),
        candidates,
        records_by_id,
    )


def family_calibration_mask(
    candidates: Sequence[Candidate],
    records_by_id: Mapping[str, QCESV5Record],
    *,
    frac: float,
) -> np.ndarray:
    families = sorted({records_by_id[item.sample_id].scene_family_id for item in candidates})
    calib_count = max(1, int(round(len(families) * frac)))
    # Stable deterministic pseudo-random split by hash.
    ranked = sorted(
        families,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    calib_families = set(ranked[:calib_count])
    return np.asarray(
        [records_by_id[item.sample_id].scene_family_id in calib_families for item in candidates],
        dtype=bool,
    )


def build_models(names: Sequence[str]) -> dict[str, Any]:
    registry = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=50,
            learning_rate=0.08,
            l2_regularization=0.03,
            max_leaf_nodes=15,
            random_state=2,
        ),
        "extra": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=2,
        ),
        "rf": RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=3,
        ),
    }
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise SystemExit("unknown models: " + ", ".join(unknown))
    return {name: registry[name] for name in names}


def scores_for(model: Any, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)


def evaluate_candidates(
    *,
    records: Sequence[QCESV5Record],
    candidates: Sequence[Candidate],
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best: dict[str, tuple[float, Candidate]] = {}
    for score, candidate in zip(scores, candidates):
        previous = best.get(candidate.sample_id)
        if previous is None or float(score) > previous[0]:
            best[candidate.sample_id] = (float(score), candidate)

    items: list[dict[str, Any]] = []
    answer_correct = answer_total = 0
    no_evidence_correct = no_evidence_total = 0
    no_evidence_decision_correct = 0
    abstain_answerable = 0
    span_iou_sum = span_iou_count = 0
    by_relation: dict[str, list[int]] = {}

    for record in records:
        selection = best.get(record.sample_id)
        if selection is None or selection[0] < threshold:
            predicted_no_evidence = True
            predicted_answer = None
            selected_events: tuple[EventProposal, ...] = ()
            score = None
        else:
            score, candidate = selection
            predicted_no_evidence = False
            predicted_answer = candidate.answer_label
            selected_events = candidate.events

        no_evidence_decision_correct += int(predicted_no_evidence == bool(record.no_evidence))
        if record.no_evidence:
            no_evidence_total += 1
            no_evidence_correct += int(predicted_no_evidence)
            answer_ok = None
            span_iou = None
        else:
            answer_total += 1
            answer_ok = predicted_answer == record.answer
            answer_correct += int(answer_ok)
            abstain_answerable += int(predicted_no_evidence)
            gold_spans = [
                (
                    float(record.event_by_id(event_id).onset_seconds),
                    float(record.event_by_id(event_id).offset_seconds),
                )
                for event_id in record.evidence_event_ids
            ]
            predicted_spans = [
                (item.onset_seconds, item.offset_seconds) for item in selected_events
            ]
            span_iou = interval_iou(predicted_spans, gold_spans)
            span_iou_sum += span_iou
            span_iou_count += 1
            by_relation.setdefault(record.relation, [0, 0])
            by_relation[record.relation][0] += int(answer_ok)
            by_relation[record.relation][1] += 1

        items.append(
            {
                "id": record.sample_id,
                "scene_id": record.scene_id,
                "scene_family_id": record.scene_family_id,
                "question": record.question,
                "relation": record.relation,
                "answer": record.answer,
                "no_evidence": bool(record.no_evidence),
                "predicted_no_evidence": predicted_no_evidence,
                "predicted_answer": predicted_answer,
                "score": score,
                "answer_correct": answer_ok,
                "span_iou_↑": span_iou,
                "selected_events": [
                    {
                        "label": item.label,
                        "onset_seconds": item.onset_seconds,
                        "offset_seconds": item.offset_seconds,
                        "confidence": item.confidence,
                    }
                    for item in selected_events
                ],
            }
        )

    summary = {
        "record_count": len(records),
        "answerable_count": answer_total,
        "no_evidence_count": no_evidence_total,
        "answer_accuracy_↑": answer_correct / answer_total if answer_total else None,
        "no_evidence_pure_accuracy_↑": (
            no_evidence_correct / no_evidence_total if no_evidence_total else None
        ),
        "no_evidence_decision_accuracy_↑": (
            no_evidence_decision_correct / len(records) if records else None
        ),
        "abstain_rate_on_answerable_↓": (
            abstain_answerable / answer_total if answer_total else None
        ),
        "span_iou_mean_↑": span_iou_sum / span_iou_count if span_iou_count else None,
        "answer_accuracy_by_relation_↑": {
            relation: value[0] / value[1] for relation, value in sorted(by_relation.items())
        },
    }
    return summary, items


def baseline_summary(
    prepared: tuple[
        list[QCESV5Record],
        dict[str, ParsedQuestion],
        dict[str, tuple[str, ...]],
        dict[str, dict[tuple[str, tuple[str, ...]], list[EventProposal]]],
    ],
) -> dict[str, Any]:
    records, parsed, labels_by_id, decoded_by_relation = prepared
    answer_correct = answer_total = 0
    no_evidence_correct = no_evidence_total = 0
    abstain = 0
    by_relation: dict[str, list[int]] = {}
    for record in records:
        question = parsed[record.sample_id]
        proposals = (
            decoded_by_relation.get(question.relation or "", {}).get(
                (record.scene_id, labels_by_id[record.sample_id]), []
            )
        )
        plan = plan_from_proposals(question, proposals)
        if record.no_evidence:
            no_evidence_total += 1
            no_evidence_correct += int(plan.no_evidence)
        else:
            answer_total += 1
            ok = plan.answer_label == record.answer
            answer_correct += int(ok)
            abstain += int(plan.no_evidence)
            by_relation.setdefault(record.relation, [0, 0])
            by_relation[record.relation][0] += int(ok)
            by_relation[record.relation][1] += 1
    return {
        "answer_accuracy_↑": answer_correct / answer_total,
        "no_evidence_pure_accuracy_↑": no_evidence_correct / no_evidence_total,
        "abstain_rate_on_answerable_↓": abstain / answer_total,
        "answer_accuracy_by_relation_↑": {
            relation: value[0] / value[1] for relation, value in sorted(by_relation.items())
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    thresholds = relation_thresholds(args.relation_threshold)

    train_prepared = prepare_split(
        manifest=args.train_manifest.resolve(),
        cache_path=args.train_cache.resolve(),
        dataset_config=args.dataset_config.resolve(),
        proposal_head=args.proposal_head.resolve(),
        thresholds=thresholds,
        device=device,
    )
    val_prepared = prepare_split(
        manifest=args.val_manifest.resolve(),
        cache_path=args.val_cache.resolve(),
        dataset_config=args.dataset_config.resolve(),
        proposal_head=args.proposal_head.resolve(),
        thresholds=thresholds,
        device=device,
    )
    train_features, train_targets, train_candidates, train_records_by_id = split_candidates(
        train_prepared, max_neighbors=args.max_neighbors
    )
    val_features, val_targets, val_candidates, _ = split_candidates(
        val_prepared, max_neighbors=args.max_neighbors
    )
    calib_mask = family_calibration_mask(
        train_candidates, train_records_by_id, frac=args.calib_family_frac
    )
    fit_mask = ~calib_mask
    if fit_mask.sum() == 0 or calib_mask.sum() == 0:
        raise SystemExit("empty fit/calibration split")

    models = build_models(args.models)
    model_reports: dict[str, Any] = {}
    best_choice: tuple[tuple[float, float, float], str, float, Any] | None = None
    train_records = train_prepared[0]
    calib_record_ids = {
        candidate.sample_id for candidate, is_calib in zip(train_candidates, calib_mask) if is_calib
    }
    calib_records = [record for record in train_records if record.sample_id in calib_record_ids]
    fit_record_ids = {
        candidate.sample_id for candidate, is_fit in zip(train_candidates, fit_mask) if is_fit
    }
    fit_records = [record for record in train_records if record.sample_id in fit_record_ids]

    for model_name, model in models.items():
        model.fit(train_features[fit_mask], train_targets[fit_mask])
        calib_scores = scores_for(model, train_features[calib_mask])
        candidate_thresholds = np.unique(
            np.r_[
                [-1e9],
                np.quantile(calib_scores, np.linspace(0.0, 0.8, 41)),
                [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            ]
        )
        best_for_model = None
        calib_candidates = [
            candidate
            for candidate, is_calib in zip(train_candidates, calib_mask)
            if is_calib
        ]
        for threshold in candidate_thresholds:
            summary, _ = evaluate_candidates(
                records=calib_records,
                candidates=calib_candidates,
                scores=calib_scores,
                threshold=float(threshold),
            )
            key = (
                float(summary["answer_accuracy_↑"]),
                float(summary["no_evidence_decision_accuracy_↑"]),
                -float(summary["abstain_rate_on_answerable_↓"]),
            )
            if best_for_model is None or key > best_for_model[0]:
                best_for_model = (key, float(threshold), summary)
        assert best_for_model is not None
        # Refit on all train candidates after choosing model/tau on train-calib.
        final_model = build_models([model_name])[model_name]
        final_model.fit(train_features, train_targets)
        train_scores = scores_for(final_model, train_features)
        val_scores = scores_for(final_model, val_features)
        train_summary, _ = evaluate_candidates(
            records=train_prepared[0],
            candidates=train_candidates,
            scores=train_scores,
            threshold=best_for_model[1],
        )
        val_summary, _ = evaluate_candidates(
            records=val_prepared[0],
            candidates=val_candidates,
            scores=val_scores,
            threshold=best_for_model[1],
        )
        model_reports[model_name] = {
            "selected_threshold": best_for_model[1],
            "calibration_summary": best_for_model[2],
            "train_summary_after_refit": train_summary,
            "val_summary": val_summary,
        }
        selection_key = (
            float(best_for_model[2]["answer_accuracy_↑"]),
            float(best_for_model[2]["no_evidence_decision_accuracy_↑"]),
            -float(best_for_model[2]["abstain_rate_on_answerable_↓"]),
        )
        if best_choice is None or selection_key > best_choice[0]:
            best_choice = (selection_key, model_name, best_for_model[1], final_model)

    assert best_choice is not None
    _, best_name, best_threshold, best_model = best_choice
    val_scores = scores_for(best_model, val_features)
    best_val_summary, best_val_items = evaluate_candidates(
        records=val_prepared[0],
        candidates=val_candidates,
        scores=val_scores,
        threshold=best_threshold,
    )

    model_path = output_dir / "pair_reranker.joblib"
    joblib.dump(
        {
            "format": FORMAT_VERSION,
            "model_name": best_name,
            "model": best_model,
            "feature_names": FEATURE_NAMES,
            "threshold": best_threshold,
            "relation_thresholds": thresholds,
            "max_neighbors": args.max_neighbors,
        },
        model_path,
    )
    (output_dir / "val_predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in best_val_items),
        encoding="utf-8",
    )

    report = {
        "format": FORMAT_VERSION,
        "device": str(device),
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "train_cache": str(args.train_cache.resolve()),
        "val_cache": str(args.val_cache.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "proposal_head": str(args.proposal_head.resolve()),
        "proposal_head_sha256": sha256_file(args.proposal_head.resolve()),
        "relation_thresholds": thresholds,
        "feature_names": FEATURE_NAMES,
        "max_neighbors": args.max_neighbors,
        "calib_family_frac": args.calib_family_frac,
        "train_candidate_count": int(train_features.shape[0]),
        "val_candidate_count": int(val_features.shape[0]),
        "train_positive_rate": float(train_targets.mean()) if train_targets.size else None,
        "val_positive_rate": float(val_targets.mean()) if val_targets.size else None,
        "baseline_relation_threshold_train": baseline_summary(train_prepared),
        "baseline_relation_threshold_val": baseline_summary(val_prepared),
        "models": model_reports,
        "selected_model": best_name,
        "selected_threshold": best_threshold,
        "selected_val_summary": best_val_summary,
        "model_path": str(model_path),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_model": best_name,
        "threshold": best_threshold,
        "baseline_val": report["baseline_relation_threshold_val"],
        "selected_val": best_val_summary,
        "output": str(output_dir),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
