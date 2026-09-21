#!/usr/bin/env python3
"""Audit QCES data before detector-first training.

This script is intentionally read-only with respect to the dataset.  It answers
the questions that matter before training a frame-level SED detector:

* how many unique audio scenes, labels, occurrences, and active seconds exist;
* whether question/evidence fields are consistent with anchor+answer evidence;
* how no-evidence examples are distributed;
* whether the current data can support a 100/120/200-class detector head;
* which labels are sufficiently covered for the first detector ontology.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_DATASET_DIR = Path("data/qces_v6_full_cropbank_v2")
DEFAULT_OUTPUT_DIR = Path("outputs/qces_detector_first_data_audit/v1")
DEFAULT_LABEL_BANK = Path("outputs/qces_open_event_v2/label_bank_200.txt")
DEFAULT_SPLITS = (
    "train",
    "val",
    "test_iid",
    "test_compositional_ood",
    "test_label_ood",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-bank", type=Path, default=DEFAULT_LABEL_BANK)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--min-scenes", type=int, default=20)
    parser.add_argument("--min-occurrences", type=int, default=60)
    parser.add_argument("--min-active-seconds", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def clean_label(label: str | None) -> str:
    return str(label or "").strip()


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def semantic_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in row.get("events", [])
        if event.get("event_kind", "semantic") == "semantic"
    ]


def event_duration(event: Mapping[str, Any]) -> float:
    return max(0.0, float(event.get("offset_seconds", 0.0)) - float(event.get("onset_seconds", 0.0)))


def event_overlap_ratio(event: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> float:
    dur = event_duration(event)
    if dur <= 0:
        return 0.0
    own_id = event.get("event_id")
    onset = float(event.get("onset_seconds", 0.0))
    offset = float(event.get("offset_seconds", 0.0))
    overlap = 0.0
    for other in events:
        if other.get("event_id") == own_id:
            continue
        overlap += interval_overlap(
            onset,
            offset,
            float(other.get("onset_seconds", 0.0)),
            float(other.get("offset_seconds", 0.0)),
        )
    return float(min(1.0, overlap / dur))


@dataclass
class LabelStats:
    label: str
    num_scenes: int
    num_occurrences: int
    total_active_seconds: float
    mean_duration_seconds: float
    median_duration_seconds: float
    mean_gain_db: float
    mean_overlap_ratio: float
    num_sources: int
    num_creators: int
    detector_ready: bool
    coverage_score: float


def load_label_bank(path: Path) -> list[str]:
    if not path.exists():
        return []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label = clean_label(line)
        if label and label not in labels:
            labels.append(label)
    return labels


def evidence_policy_status(row: Mapping[str, Any]) -> str:
    evidence = set(row.get("evidence_event_ids") or [])
    anchors = set(row.get("anchor_event_ids") or [])
    answers = set(row.get("answer_event_ids") or [])
    no_evidence = bool(row.get("no_evidence", False))

    if no_evidence:
        if not evidence and not anchors and not answers:
            return "no_evidence_empty_ok"
        return "no_evidence_has_events_bad"

    expected = anchors | answers
    if not evidence:
        return "positive_empty_bad"
    if evidence == expected and evidence:
        if anchors and answers:
            return "anchor_plus_answer_ok"
        if answers and not anchors:
            return "answer_only_expected_ok"
        if anchors and not answers:
            return "anchor_only_expected_ok"
        return "positive_other_ok"
    if evidence == answers and answers:
        return "answer_only_but_anchor_expected"
    if evidence == anchors and anchors:
        return "anchor_only_but_answer_expected"
    if expected and expected.issubset(evidence):
        return "has_anchor_answer_plus_extra"
    return "mismatch_bad"


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def format_count_table(counter: Mapping[str, int], *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [{"name": key, "count": value} for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]
    return rows[:limit] if limit else rows


def compute_label_stats(
    scenes: Mapping[str, Mapping[str, Any]],
    *,
    min_scenes: int,
    min_occurrences: int,
    min_active_seconds: float,
) -> list[LabelStats]:
    scene_ids_by_label: dict[str, set[str]] = defaultdict(set)
    durations_by_label: dict[str, list[float]] = defaultdict(list)
    gain_by_label: dict[str, list[float]] = defaultdict(list)
    overlap_by_label: dict[str, list[float]] = defaultdict(list)
    sources_by_label: dict[str, set[str]] = defaultdict(set)
    creators_by_label: dict[str, set[str]] = defaultdict(set)

    for scene_id, row in scenes.items():
        events = semantic_events(row)
        for event in events:
            label = clean_label(event.get("label"))
            if not label:
                continue
            scene_ids_by_label[label].add(scene_id)
            durations_by_label[label].append(event_duration(event))
            if event.get("gain_db") is not None:
                try:
                    gain_by_label[label].append(float(event["gain_db"]))
                except (TypeError, ValueError):
                    pass
            overlap_by_label[label].append(event_overlap_ratio(event, events))
            if event.get("source_id") is not None:
                sources_by_label[label].add(str(event["source_id"]))
            if event.get("creator_id") is not None:
                creators_by_label[label].add(str(event["creator_id"]))

    stats: list[LabelStats] = []
    for label in sorted(durations_by_label):
        durations = durations_by_label[label]
        num_scenes = len(scene_ids_by_label[label])
        num_occurrences = len(durations)
        total_active = float(sum(durations))
        detector_ready = (
            num_scenes >= min_scenes
            and num_occurrences >= min_occurrences
            and total_active >= min_active_seconds
        )
        coverage_score = (
            min(num_scenes / max(min_scenes, 1), 1.0)
            + min(num_occurrences / max(min_occurrences, 1), 1.0)
            + min(total_active / max(min_active_seconds, 1e-6), 1.0)
        ) / 3.0
        stats.append(
            LabelStats(
                label=label,
                num_scenes=num_scenes,
                num_occurrences=num_occurrences,
                total_active_seconds=total_active,
                mean_duration_seconds=mean(durations),
                median_duration_seconds=median(durations),
                mean_gain_db=mean(gain_by_label[label]),
                mean_overlap_ratio=mean(overlap_by_label[label]),
                num_sources=len(sources_by_label[label]),
                num_creators=len(creators_by_label[label]),
                detector_ready=detector_ready,
                coverage_score=coverage_score,
            )
        )
    stats.sort(key=lambda item: (-item.coverage_score, -item.num_scenes, -item.num_occurrences, item.label))
    return stats


def audit_split(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    scenes: dict[str, dict[str, Any]] = {}
    rows_per_scene: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    no_evidence_counts: Counter[str] = Counter()
    evidence_status_counts: Counter[str] = Counter()
    hard_case_counts: Counter[str] = Counter()
    schema_versions: Counter[str] = Counter()
    inconsistent_scene_event_signatures: list[str] = []
    scene_event_signature: dict[str, tuple[tuple[str, str, float, float], ...]] = {}

    for row in rows:
        scene_id = str(row.get("scene_id") or "")
        rows_per_scene[scene_id] += 1
        question_type_counts[clean_label(row.get("question_type")) or "unknown"] += 1
        relation_counts[clean_label(row.get("relation")) or "none"] += 1
        schema_versions[clean_label(row.get("schema_version")) or "unknown"] += 1
        evidence_status_counts[evidence_policy_status(row)] += 1
        if row.get("no_evidence"):
            no_evidence_counts[clean_label(row.get("no_evidence_reason")) or "unspecified"] += 1
        for tag in row.get("hard_case_tags") or []:
            hard_case_counts[str(tag)] += 1

        events = semantic_events(row)
        signature = tuple(
            sorted(
                (
                    str(event.get("event_id")),
                    clean_label(event.get("label")),
                    round(float(event.get("onset_seconds", 0.0)), 6),
                    round(float(event.get("offset_seconds", 0.0)), 6),
                )
                for event in events
            )
        )
        if scene_id not in scenes:
            scenes[scene_id] = row
            scene_event_signature[scene_id] = signature
        elif scene_event_signature.get(scene_id) != signature and len(inconsistent_scene_event_signatures) < 20:
            inconsistent_scene_event_signatures.append(scene_id)

    event_counts = [len(semantic_events(row)) for row in scenes.values()]
    label_counts = [len({clean_label(event.get("label")) for event in semantic_events(row)}) for row in scenes.values()]
    row_counts_per_scene = list(rows_per_scene.values())
    max_polyphony_values = [float(row.get("max_polyphony") or 0.0) for row in scenes.values()]
    semantic_overlap_values = [float(row.get("semantic_overlap") or 0.0) for row in scenes.values()]

    return {
        "path": str(path),
        "rows": len(rows),
        "unique_scenes": len(scenes),
        "scene_families": len({row.get("scene_family_id") for row in scenes.values()}),
        "schema_versions": dict(schema_versions),
        "rows_per_scene": summarize_values([float(value) for value in row_counts_per_scene]),
        "event_count_per_scene": summarize_values([float(value) for value in event_counts]),
        "unique_label_count_per_scene": summarize_values([float(value) for value in label_counts]),
        "max_polyphony": summarize_values(max_polyphony_values),
        "semantic_overlap": summarize_values(semantic_overlap_values),
        "question_type_counts": dict(question_type_counts),
        "relation_counts": dict(relation_counts),
        "no_evidence_reason_counts": dict(no_evidence_counts),
        "evidence_policy_status_counts": dict(evidence_status_counts),
        "hard_case_counts": dict(hard_case_counts),
        "inconsistent_scene_event_signatures": inconsistent_scene_event_signatures,
        "scenes": scenes,
    }


def detector_scene_manifest_rows(scenes: Mapping[str, Mapping[str, Any]], label_to_id: Mapping[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_id, row in sorted(scenes.items()):
        events = []
        for event in semantic_events(row):
            label = clean_label(event.get("label"))
            if label not in label_to_id:
                continue
            events.append(
                {
                    "event_id": event.get("event_id"),
                    "label": label,
                    "label_id": label_to_id[label],
                    "onset_seconds": float(event.get("onset_seconds", 0.0)),
                    "offset_seconds": float(event.get("offset_seconds", 0.0)),
                    "duration_seconds": event_duration(event),
                    "source_id": event.get("source_id"),
                    "source_path": event.get("source_path"),
                    "stem_path": event.get("stem_path"),
                }
            )
        rows.append(
            {
                "scene_id": scene_id,
                "split": row.get("split"),
                "scene_family_id": row.get("scene_family_id"),
                "mixture_path": row.get("mixture_path"),
                "duration_seconds": row.get("duration_seconds"),
                "sample_rate": row.get("sample_rate"),
                "events": events,
            }
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def markdown_table(rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(str(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_bank = load_label_bank((Path.cwd() / args.label_bank).resolve() if not args.label_bank.is_absolute() else args.label_bank)
    split_reports: dict[str, dict[str, Any]] = {}
    all_scenes: dict[str, dict[str, Any]] = {}
    train_scenes: dict[str, dict[str, Any]] = {}

    for split in args.splits:
        path = dataset_dir / f"qces_{split}.jsonl"
        if not path.exists():
            continue
        report = audit_split(path)
        scenes = report.pop("scenes")
        split_reports[split] = report
        for scene_id, row in scenes.items():
            all_scenes[f"{split}:{scene_id}"] = row
        if split == "train":
            train_scenes = scenes

    train_label_stats = compute_label_stats(
        train_scenes,
        min_scenes=args.min_scenes,
        min_occurrences=args.min_occurrences,
        min_active_seconds=args.min_active_seconds,
    )
    all_label_stats = compute_label_stats(
        all_scenes,
        min_scenes=args.min_scenes,
        min_occurrences=args.min_occurrences,
        min_active_seconds=args.min_active_seconds,
    )

    train_labels = {item.label for item in train_label_stats}
    all_labels = {item.label for item in all_label_stats}
    label_bank_set = set(label_bank)
    covered_bank_train = sorted(label_bank_set & train_labels)
    covered_bank_all = sorted(label_bank_set & all_labels)
    bank_without_positive_train = sorted(label_bank_set - train_labels)
    bank_without_positive_all = sorted(label_bank_set - all_labels)

    ready_train = [item for item in train_label_stats if item.detector_ready]
    ontology_top100 = train_label_stats[:100]
    ontology_top120 = train_label_stats[:120]
    ontology_ready = ready_train

    label_to_id = {item.label: index for index, item in enumerate(train_label_stats)}
    for split in args.splits:
        path = dataset_dir / f"qces_{split}.jsonl"
        if not path.exists():
            continue
        split_audit = audit_split(path)
        scenes = split_audit["scenes"]
        write_jsonl(
            output_dir / f"detector_scene_manifest_{split}.jsonl",
            detector_scene_manifest_rows(scenes, label_to_id),
        )

    write_jsonl(output_dir / "label_stats_train.jsonl", [asdict(item) for item in train_label_stats])
    write_jsonl(output_dir / "label_stats_all_splits.jsonl", [asdict(item) for item in all_label_stats])
    (output_dir / "ontology_top100_train.txt").write_text("\n".join(item.label for item in ontology_top100) + "\n", encoding="utf-8")
    (output_dir / "ontology_top120_train.txt").write_text("\n".join(item.label for item in ontology_top120) + "\n", encoding="utf-8")
    (output_dir / "ontology_ready_train.txt").write_text("\n".join(item.label for item in ontology_ready) + "\n", encoding="utf-8")

    summary = {
        "format": "qces_detector_first_data_audit",
        "dataset_dir": str(dataset_dir),
        "splits": split_reports,
        "label_bank_path": str(args.label_bank),
        "label_bank_size": len(label_bank),
        "label_bank_covered_by_train_positive": len(covered_bank_train),
        "label_bank_covered_by_any_split_positive": len(covered_bank_all),
        "label_bank_without_train_positive_count": len(bank_without_positive_train),
        "label_bank_without_any_positive_count": len(bank_without_positive_all),
        "qces_train_positive_label_count": len(train_labels),
        "qces_all_positive_label_count": len(all_labels),
        "detector_ready_criteria": {
            "min_scenes": args.min_scenes,
            "min_occurrences": args.min_occurrences,
            "min_active_seconds": args.min_active_seconds,
        },
        "detector_ready_train_label_count": len(ready_train),
        "top_train_labels": [asdict(item) for item in train_label_stats[:30]],
        "low_coverage_train_labels": [asdict(item) for item in sorted(train_label_stats, key=lambda item: (item.coverage_score, item.num_scenes, item.num_occurrences, item.label))[:30]],
        "covered_bank_train_labels": covered_bank_train,
        "covered_bank_all_labels": covered_bank_all,
        "bank_without_positive_train_sample": bank_without_positive_train[:80],
        "bank_without_positive_all_sample": bank_without_positive_all[:80],
    }
    write_json(output_dir / "detector_first_data_audit.json", summary)

    split_rows = [["split", "rows", "unique scenes", "scene families", "rows/scene mean", "labels/scene mean"]]
    for split, report in split_reports.items():
        split_rows.append(
            [
                split,
                report["rows"],
                report["unique_scenes"],
                report["scene_families"],
                f"{report['rows_per_scene']['mean']:.1f}",
                f"{report['unique_label_count_per_scene']['mean']:.1f}",
            ]
        )

    relation_counter = Counter()
    qtype_counter = Counter()
    noev_counter = Counter()
    evidence_counter = Counter()
    for report in split_reports.values():
        relation_counter.update(report["relation_counts"])
        qtype_counter.update(report["question_type_counts"])
        noev_counter.update(report["no_evidence_reason_counts"])
        evidence_counter.update(report["evidence_policy_status_counts"])

    top_label_rows = [["label", "scenes", "occ", "active sec", "mean dur", "mean overlap", "ready"]]
    for item in train_label_stats[:30]:
        top_label_rows.append(
            [
                item.label,
                item.num_scenes,
                item.num_occurrences,
                f"{item.total_active_seconds:.1f}",
                f"{item.mean_duration_seconds:.2f}",
                f"{item.mean_overlap_ratio:.2f}",
                "yes" if item.detector_ready else "no",
            ]
        )

    low_label_rows = [["label", "scenes", "occ", "active sec", "coverage score"]]
    for item in sorted(train_label_stats, key=lambda item: (item.coverage_score, item.num_scenes, item.num_occurrences, item.label))[:20]:
        low_label_rows.append(
            [
                item.label,
                item.num_scenes,
                item.num_occurrences,
                f"{item.total_active_seconds:.1f}",
                f"{item.coverage_score:.2f}",
            ]
        )

    md_lines = [
        "# QCES detector-first data audit",
        "",
        "## Dataset scale",
        "",
        markdown_table(split_rows),
        "",
        "## Main finding",
        "",
        (
            f"- QCES train has **{len(train_labels)}** positive labels. "
            f"The provided label bank has **{len(label_bank)}** labels, but only "
            f"**{len(covered_bank_train)}** have positive training examples."
        ),
        (
            f"- Across all splits, QCES has **{len(all_labels)}** positive labels; "
            f"**{len(bank_without_positive_all)}** labels from the bank have no positive example anywhere in this dataset."
        ),
        (
            "- Therefore, a supervised 200-class detector head is not supported by the current QCES data unless the dataset is rebuilt or augmented with positive examples for the missing classes."
        ),
        "",
        "## Evidence policy check",
        "",
        markdown_table([["status", "count"]] + [[row["name"], row["count"]] for row in format_count_table(evidence_counter)]),
        "",
        "## Question/relation balance",
        "",
        markdown_table([["relation", "count"]] + [[row["name"], row["count"]] for row in format_count_table(relation_counter)]),
        "",
        markdown_table([["question_type", "count"]] + [[row["name"], row["count"]] for row in format_count_table(qtype_counter)]),
        "",
        "## No-evidence reasons",
        "",
        markdown_table([["reason", "count"]] + [[row["name"], row["count"]] for row in format_count_table(noev_counter)]),
        "",
        "## Top train label coverage",
        "",
        markdown_table(top_label_rows),
        "",
        "## Lowest train label coverage",
        "",
        markdown_table(low_label_rows),
        "",
        "## Detector ontology recommendation",
        "",
        f"- `ontology_ready_train.txt`: {len(ontology_ready)} labels passing current readiness thresholds.",
        f"- `ontology_top100_train.txt`: {len(ontology_top100)} labels available from train, but current train only has {len(train_labels)} unique labels if fewer than 100.",
        f"- `ontology_top120_train.txt`: {len(ontology_top120)} labels available from train, but current train only has {len(train_labels)} unique labels if fewer than 120.",
        "",
        "Recommended immediate detector experiment:",
        "",
        "```text",
        "Train PretrainedSED head on the current train-positive ontology first, not 200 classes.",
        "Use the 200-label bank only for zero-shot baselines such as FlexSED/FLAM, or rebuild QCES with positive examples for the missing labels.",
        "```",
        "",
        "## Generated files",
        "",
        "- `detector_first_data_audit.json`",
        "- `label_stats_train.jsonl`",
        "- `label_stats_all_splits.jsonl`",
        "- `ontology_ready_train.txt`",
        "- `ontology_top100_train.txt`",
        "- `ontology_top120_train.txt`",
        "- `detector_scene_manifest_<split>.jsonl`",
        "",
    ]
    (output_dir / "detector_first_data_audit.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
