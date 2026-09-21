#!/usr/bin/env python3
"""Audit QCES multiple-choice leakage without loading a waveform.

The benchmark deliberately includes binary ``first(A, B)`` questions in a
five-option format.  A language-only system can narrow those questions to two
named candidates, so uniform five-way chance (20%) is the wrong reference.
This audit reports both empirical lookup baselines and the relation-aware
candidate chance floor (50% for ``first``, 20% otherwise).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from mixi_understanding.scripts.evaluate_qces_shortcuts import normalize_question


ACCURACY = "multiple_choice_accuracy_↑"
NO_EVIDENCE_RECALL = "no_evidence_recall_↑"
ANSWERABLE_ACCURACY = "answerable_accuracy_↑"
ANSWERABLE_BINARY_RECALL = "answerable_binary_recall_↑"
NO_EVIDENCE_FALSE_POSITIVE_RATE = "no_evidence_false_positive_rate_↓"
NO_EVIDENCE_BALANCED_ACCURACY = "no_evidence_balanced_accuracy_↑"
SEEN_KEY_RATE = "held_out_key_seen_rate_↑"
EXCESS_OVER_CANDIDATE_CHANCE = "excess_over_candidate_aware_chance_↓"


@dataclass(frozen=True)
class QAExample:
    record_id: str
    scene_id: str
    split: str
    question: str
    relation: str
    query_labels: Tuple[str, ...]
    answer: str
    options: Tuple[str, ...]
    answer_index: int

    @property
    def no_evidence(self) -> bool:
        return self.answer == "no_evidence"


@dataclass(frozen=True)
class Prediction:
    example: QAExample
    answer: str
    key_seen: bool

    @property
    def correct(self) -> bool:
        return self.answer == self.example.answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--include-val-in-train",
        action="store_true",
        help="Fit on train+val before the one-shot test audit.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if args.include_val_in_train and args.eval_split != "test":
        parser.error("--include-val-in-train is only valid for test")
    return args


def _text_list(value: object, context: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    result = tuple(str(item) for item in value)
    if any(not item.strip() for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{context} must contain unique non-empty strings")
    return result


def load_examples(path: Path) -> List[QAExample]:
    examples: List[QAExample] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or payload.get("schema_version") != "qces_v4":
                raise ValueError(f"{path}:{line_number} is not a qces_v4 record")
            record_id = str(payload.get("id", ""))
            if not record_id or record_id in seen_ids:
                raise ValueError(f"invalid/duplicate record id at line {line_number}")
            seen_ids.add(record_id)
            options = _text_list(payload.get("answer_options"), "answer_options")
            query_labels = _text_list(payload.get("query_labels"), "query_labels")
            answer = str(payload.get("answer", ""))
            answer_index = payload.get("answer_option_index")
            if not isinstance(answer_index, int) or not 0 <= answer_index < len(options):
                raise ValueError(f"invalid answer index at line {line_number}")
            if options[answer_index] != answer:
                raise ValueError(f"answer/index mismatch at line {line_number}")
            relation = str(payload.get("relation", ""))
            if relation not in {"after", "before", "first"}:
                raise ValueError(f"invalid relation at line {line_number}")
            if relation == "first" and not set(query_labels).issubset(options):
                raise ValueError(f"first candidates absent from options at line {line_number}")
            examples.append(
                QAExample(
                    record_id=record_id,
                    scene_id=str(payload.get("scene_id", "")),
                    split=str(payload.get("split", "")),
                    question=normalize_question(str(payload.get("question", ""))),
                    relation=relation,
                    query_labels=query_labels,
                    answer=answer,
                    options=options,
                    answer_index=answer_index,
                )
            )
    if not examples:
        raise ValueError(f"empty manifest: {path}")
    return examples


def _mode(counter: Mapping[str, int], order: Sequence[str]) -> str:
    return min(order, key=lambda label: (-int(counter.get(label, 0)), order.index(label)))


def _fit_lookup(
    rows: Sequence[QAExample], key_fn: Callable[[QAExample], str]
) -> Dict[str, Counter[str]]:
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[key_fn(row)][row.answer] += 1
    return counts


def _fallback_answer(row: QAExample, global_answers: Counter[str], position: int) -> str:
    compatible = {label: count for label, count in global_answers.items() if label in row.options}
    if compatible:
        return _mode(compatible, row.options)
    return row.options[position % len(row.options)]


def lookup_predictions(
    train: Sequence[QAExample],
    evaluation: Sequence[QAExample],
    key_fn: Callable[[QAExample], str],
) -> List[Prediction]:
    lookup = _fit_lookup(train, key_fn)
    global_answers = Counter(row.answer for row in train)
    position_counts = Counter(row.answer_index for row in train)
    fallback_position = min(range(5), key=lambda index: (-position_counts[index], index))
    predictions: List[Prediction] = []
    for row in evaluation:
        key = key_fn(row)
        counts = lookup.get(key)
        compatible = (
            {label: count for label, count in counts.items() if label in row.options}
            if counts
            else {}
        )
        if compatible:
            answer = _mode(compatible, row.options)
            seen = True
        else:
            answer = _fallback_answer(row, global_answers, fallback_position)
            seen = False
        predictions.append(Prediction(row, answer, seen))
    return predictions


def position_prior_predictions(
    train: Sequence[QAExample], evaluation: Sequence[QAExample]
) -> List[Prediction]:
    counts = Counter(row.answer_index for row in train)
    index = min(range(5), key=lambda value: (-counts[value], value))
    return [Prediction(row, row.options[index], True) for row in evaluation]


def named_candidate_predictions(
    evaluation: Sequence[QAExample], candidate_index: int
) -> List[Prediction]:
    predictions = []
    for row in evaluation:
        if row.relation == "first":
            answer = row.query_labels[candidate_index]
        else:
            digest = hashlib.sha256(row.record_id.encode("utf-8")).digest()
            answer = row.options[int.from_bytes(digest[:4], "big") % len(row.options)]
        predictions.append(Prediction(row, answer, True))
    return predictions


def _safe_mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _candidate_chance(rows: Sequence[QAExample]) -> float:
    return sum(0.5 if row.relation == "first" else 0.2 for row in rows) / len(rows)


def summarize(predictions: Sequence[Prediction]) -> Dict[str, object]:
    rows = [prediction.example for prediction in predictions]
    accuracy = _safe_mean(float(prediction.correct) for prediction in predictions)
    no_evidence = [prediction for prediction in predictions if prediction.example.no_evidence]
    answerable = [prediction for prediction in predictions if not prediction.example.no_evidence]
    candidate_chance = _candidate_chance(rows)
    no_evidence_recall = _safe_mean(float(item.correct) for item in no_evidence)
    answerable_binary_recall = _safe_mean(
        float(item.answer != "no_evidence") for item in answerable
    )
    no_evidence_balanced_accuracy = (
        None
        if no_evidence_recall is None or answerable_binary_recall is None
        else (no_evidence_recall + answerable_binary_recall) / 2.0
    )
    return {
        ACCURACY: accuracy,
        ANSWERABLE_ACCURACY: _safe_mean(float(item.correct) for item in answerable),
        NO_EVIDENCE_RECALL: no_evidence_recall,
        ANSWERABLE_BINARY_RECALL: answerable_binary_recall,
        NO_EVIDENCE_FALSE_POSITIVE_RATE: (
            None
            if answerable_binary_recall is None
            else 1.0 - answerable_binary_recall
        ),
        NO_EVIDENCE_BALANCED_ACCURACY: no_evidence_balanced_accuracy,
        SEEN_KEY_RATE: _safe_mean(float(item.key_seen) for item in predictions),
        "candidate_aware_chance_accuracy_↑": candidate_chance,
        EXCESS_OVER_CANDIDATE_CHANCE: float(accuracy) - candidate_chance,
        "accuracy_by_relation_↑": {
            relation: _safe_mean(
                float(item.correct)
                for item in predictions
                if item.example.relation == relation
            )
            for relation in ("after", "before", "first")
        },
    }


def scene_bootstrap_ci(
    predictions: Sequence[Prediction], samples: int, seed: int
) -> List[float]:
    grouped: Dict[str, List[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.example.scene_id].append(prediction)
    scene_ids = sorted(grouped)
    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(samples):
        draw = [rng.choice(scene_ids) for _ in scene_ids]
        selected = [item for scene_id in draw for item in grouped[scene_id]]
        estimates.append(sum(item.correct for item in selected) / len(selected))
    estimates.sort()
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [lower, upper]


def main() -> None:
    args = parse_args()
    rows = load_examples(args.manifest)
    train_splits = {"train"}
    if args.include_val_in_train:
        train_splits.add("val")
    train = [row for row in rows if row.split in train_splits]
    evaluation = [row for row in rows if row.split == args.eval_split]
    if not train or not evaluation:
        raise ValueError("manifest does not contain the requested train/eval splits")

    systems = {
        "answer_option_position_prior": position_prior_predictions(train, evaluation),
        "relation_plus_query_labels_lookup": lookup_predictions(
            train,
            evaluation,
            lambda row: row.relation + "|" + "|".join(sorted(row.query_labels)),
        ),
        "normalized_exact_question_lookup": lookup_predictions(
            train, evaluation, lambda row: row.question
        ),
        "first_named_candidate_0": named_candidate_predictions(evaluation, 0),
        "first_named_candidate_1": named_candidate_predictions(evaluation, 1),
    }
    summaries = {}
    for index, (name, predictions) in enumerate(systems.items()):
        summary = summarize(predictions)
        summary["scene_bootstrap_95ci_accuracy_↑"] = scene_bootstrap_ci(
            predictions, args.bootstrap_samples, args.seed + index
        )
        summaries[name] = summary

    position_counts = Counter(row.answer_index for row in rows)
    first_rows = [row for row in rows if row.relation == "first"]
    first_correct_position = Counter(
        row.query_labels.index(row.answer) for row in first_rows
    )
    report = {
        "format": "qces_v4_no_audio_qa_shortcut_audit_v1",
        "manifest": str(args.manifest.resolve()),
        "training_splits": sorted(train_splits),
        "evaluation_split": args.eval_split,
        "counts": {"train": len(train), "evaluation": len(evaluation)},
        "baselines": summaries,
        "design_diagnostics": {
            "answer_option_position_counts": dict(sorted(position_counts.items())),
            "answer_option_position_max_deviation_from_uniform_↓": max(
                abs(position_counts[index] / len(rows) - 0.2) for index in range(5)
            ),
            "first_answer_query_position_counts": dict(
                sorted(first_correct_position.items())
            ),
            "first_answer_query_position_absolute_bias_↓": abs(
                first_correct_position[0] / len(first_rows) - 0.5
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("No-audio multiple-choice shortcut baselines")
    print(f"{'Baseline':40s} {'Accuracy ↑':>11s} {'Excess ↓':>10s} {'NE recall ↑':>12s}")
    for name, summary in summaries.items():
        no_evidence_recall = summary[NO_EVIDENCE_RECALL]
        no_evidence_text = "n/a" if no_evidence_recall is None else f"{no_evidence_recall:.4f}"
        print(
            f"{name:40s} {summary[ACCURACY]:11.4f} "
            f"{summary[EXCESS_OVER_CANDIDATE_CHANCE]:10.4f} "
            f"{no_evidence_text:>12s}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
