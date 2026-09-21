#!/usr/bin/env python3
"""Select a clean, diverse, scene-disjoint 300-record QCES-v5 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


VARIANTS = ("base", "order_swap", "anchor_drop")
RELATION_QUOTAS = {"after": 34, "before": 33, "first": 33}
NEGATIVE_GROUP_QUOTAS = {"after": 8, "before": 8, "first": 8}
MINIMUM_SCENE_CROP_PROBABILITY = 0.001


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--crop-bank", type=Path, required=True)
    parser.add_argument("--weak-listening-queue", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-val", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-selection",
        type=Path,
        help=(
            "Receipt from a planning run. Reuse its exact family/question "
            "groups instead of ranking the input manifest again."
        ),
    )
    parser.add_argument(
        "--family-index-template",
        type=Path,
        help="Debug-family index-list JSON whose non-train indices are preserved.",
    )
    parser.add_argument(
        "--family-index-output",
        type=Path,
        help=(
            "Write a builder-compatible debug-family index list containing "
            "the selected train families."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if (args.family_index_template is None) != (args.family_index_output is None):
        parser.error(
            "--family-index-template and --family-index-output must be used together"
        )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def source_quality(
    rows: Sequence[Mapping[str, Any]],
    bank: Mapping[tuple[str, str], Mapping[str, Any]],
    weak: set[tuple[str, str]],
    *,
    reject_weak: bool,
) -> dict[str, Any]:
    keys = {
        (str(event["source_id"]), str(event["label"]))
        for row in rows
        for event in row.get("events", [])
        if event.get("event_kind") == "semantic"
    }
    missing = sorted(key for key in keys if key not in bank)
    overlap = sorted(keys & weak)
    if missing:
        raise ValueError(f"selected rows missing crop-bank entries: {missing}")
    if reject_weak and overlap:
        raise ValueError(f"selected rows overlap weak listening queue: {overlap}")
    scores = [
        float(bank[key]["selected"]["label_probability ↑"])
        for key in sorted(keys)
    ]
    return {
        "unique_semantic_sources ↑": len(keys),
        "weakest_queue_overlap ↓": len(overlap),
        "minimum_crop_probability ↑": min(scores),
        "mean_crop_probability ↑": statistics.mean(scores),
        "median_crop_probability ↑": statistics.median(scores),
    }


def check_complete_groups(rows: Sequence[Mapping[str, Any]]) -> None:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scene_family_id"]), int(row["question_index"]))].append(row)
    if len(rows) != len(groups) * 3:
        raise ValueError("pilot selection must contain complete 3-variant groups")
    for key, group in groups.items():
        if {str(row["variant_id"]) for row in group} != set(VARIANTS):
            raise ValueError(f"incomplete variant group: {key}")


def semantic_source_keys(
    rows: Sequence[Mapping[str, Any]],
    *,
    evidence_only: bool,
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        evidence_ids = {str(value) for value in row.get("evidence_event_ids", [])}
        for event in row.get("events", []):
            if event.get("event_kind") != "semantic":
                continue
            if evidence_only and str(event["event_id"]) not in evidence_ids:
                continue
            keys.add((str(event["source_id"]), str(event["label"])))
    return keys


def probability_stats(
    keys: set[tuple[str, str]],
    bank: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[float, float]:
    missing = sorted(key for key in keys if key not in bank)
    if missing:
        raise ValueError(f"candidate group missing crop-bank entries: {missing}")
    probabilities = [
        float(bank[key]["selected"]["label_probability ↑"]) for key in sorted(keys)
    ]
    if not probabilities:
        return 0.0, 0.0
    return min(probabilities), statistics.mean(probabilities)


def rank_group(
    key: tuple[str, int],
    rows: Sequence[Mapping[str, Any]],
    bank: Mapping[tuple[str, str], Mapping[str, Any]],
    weak: set[tuple[str, str]],
) -> dict[str, Any]:
    if len(rows) != 3 or {str(row["variant_id"]) for row in rows} != set(VARIANTS):
        raise ValueError(f"incomplete candidate group: {key}")
    relations = {str(row["relation"]) for row in rows}
    if len(relations) != 1:
        raise ValueError(f"candidate group changes relation: {key}")
    evidence_keys = semantic_source_keys(rows, evidence_only=True)
    scene_keys = semantic_source_keys(rows, evidence_only=False)
    target_minimum, target_mean = probability_stats(evidence_keys, bank)
    scene_minimum, scene_mean = probability_stats(scene_keys, bank)
    no_evidence_count = sum(bool(row["no_evidence"]) for row in rows)
    answerable_count = len(rows) - no_evidence_count
    target_weak = len(evidence_keys & weak)
    scene_weak = len(scene_keys & weak)
    # The target is the supervised signal, so it must avoid every source in
    # the manual-listening queue. Background sources remain a soft ranking
    # term: they are uncertain, not confirmed failures.
    all_negative = answerable_count == 0
    eligible = (
        ((answerable_count >= 2 and target_weak == 0) or all_negative)
        and scene_minimum >= MINIMUM_SCENE_CROP_PROBABILITY
    )
    if all_negative:
        score = (
            -scene_weak,
            scene_minimum,
            scene_mean,
            -key[1],
        )
    else:
        score = (
            -abs(no_evidence_count - 1),
            target_minimum,
            target_mean,
            -scene_weak,
            scene_minimum,
            scene_mean,
            -key[1],
        )
    return {
        "family_id": key[0],
        "question_index": key[1],
        "relation": next(iter(relations)),
        "answerable_records ↑": answerable_count,
        "no_evidence_records ↑": no_evidence_count,
        "target_unique_sources ↑": len(evidence_keys),
        "target_weak_queue_overlap ↓": target_weak,
        "scene_weak_queue_overlap ↓": scene_weak,
        "target_minimum_crop_probability ↑": target_minimum,
        "target_mean_crop_probability ↑": target_mean,
        "scene_minimum_crop_probability ↑": scene_minimum,
        "scene_mean_crop_probability ↑": scene_mean,
        "_all_negative": all_negative,
        "_eligible": eligible,
        "_score": score,
    }


def select_diverse_groups(
    by_group: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    bank: Mapping[tuple[str, str], Mapping[str, Any]],
    weak: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    candidates = [
        rank_group(key, rows, bank, weak) for key, rows in sorted(by_group.items())
    ]
    best_by_family_bucket: dict[tuple[str, str, bool], dict[str, Any]] = {}
    for candidate in candidates:
        if not candidate["_eligible"]:
            continue
        family_bucket = (
            str(candidate["family_id"]),
            str(candidate["relation"]),
            bool(candidate["_all_negative"]),
        )
        previous = best_by_family_bucket.get(family_bucket)
        if previous is None or candidate["_score"] > previous["_score"]:
            best_by_family_bucket[family_bucket] = candidate

    category_quotas: dict[tuple[str, bool], int] = {}
    for relation, total in RELATION_QUOTAS.items():
        negative = NEGATIVE_GROUP_QUOTAS[relation]
        category_quotas[(relation, True)] = negative
        category_quotas[(relation, False)] = total - negative
    ranked_by_category: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for category in category_quotas:
        relation, all_negative = category
        ranked_by_category[category] = sorted(
            (
                candidate
                for (family_id, candidate_relation, candidate_negative), candidate
                in best_by_family_bucket.items()
                if candidate_relation == relation
                and candidate_negative == all_negative
            ),
            key=lambda candidate: (
                candidate["_score"],
                str(candidate["family_id"]),
            ),
            reverse=True,
        )

    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    cursors = {category: 0 for category in category_quotas}
    remaining = dict(category_quotas)
    while any(count > 0 for count in remaining.values()):
        progressed = False
        for category in category_quotas:
            if remaining[category] == 0:
                continue
            ranked = ranked_by_category[category]
            while cursors[category] < len(ranked):
                candidate = ranked[cursors[category]]
                cursors[category] += 1
                family_id = str(candidate["family_id"])
                if family_id in used_families:
                    continue
                selected.append(candidate)
                used_families.add(family_id)
                remaining[category] -= 1
                progressed = True
                break
        if not progressed:
            raise ValueError(
                "not enough distinct clean-target families for relation quotas: "
                f"{remaining}"
            )
    if len(selected) != 100 or len(used_families) != 100:
        raise AssertionError("diverse pilot must select 100 distinct train families")
    return selected


def read_expected_selection(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    selected = payload.get("selection", {}).get("selected_groups")
    if not isinstance(selected, list) or len(selected) != 100:
        raise ValueError(
            "expected-selection receipt must contain 100 selected_groups"
        )
    return [dict(item) for item in selected]


def serializable_group(group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in group.items()
        if not str(key).startswith("_")
    }


def write_family_index_list(
    *,
    template_path: Path,
    output_path: Path,
    selected_groups: Sequence[Mapping[str, Any]],
    overwrite: bool,
) -> None:
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_path}; use --overwrite")
    payload = json.loads(template_path.resolve().read_text(encoding="utf-8"))
    indices = payload.get("indices")
    if not isinstance(indices, dict):
        raise ValueError("family-index template is missing indices")
    train_indices = sorted(
        int(str(group["family_id"]).rsplit("_", 1)[1])
        for group in selected_groups
    )
    indices["train"] = train_indices
    payload["purpose"] = (
        "100-family clean-target pilot300; one complete three-variant "
        "question group per train family; not paper evidence"
    )
    payload["selection_rule"] = {
        "records": "100 distinct train families x one question group x 3 variants",
        "target_gate": "zero overlap with crop-bank v2 manual-listening queue",
        "scene_floor": {
            "metric": "minimum selected BEATs label probability",
            "value": MINIMUM_SCENE_CROP_PROBABILITY,
        },
        "relation_quotas": RELATION_QUOTAS,
        "all_negative_group_quotas": NEGATIVE_GROUP_QUOTAS,
        "ranking": (
            "target minimum/mean crop probability, then background weak-source "
            "count and scene crop probability"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train_rows = read_jsonl(args.train_manifest.resolve())
    val_rows = read_jsonl(args.val_manifest.resolve())
    by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_group[(str(row["scene_family_id"]), int(row["question_index"]))].append(row)

    crop_payload = json.loads(args.crop_bank.resolve().read_text(encoding="utf-8"))
    bank = {
        (str(entry["source_id"]), str(entry["label"])): entry
        for entry in crop_payload["entries"]
    }
    weak = {
        (str(row["source_id"]), str(row["label"]))
        for row in read_jsonl(args.weak_listening_queue.resolve())
    }
    if args.expected_selection is None:
        selected_groups = select_diverse_groups(by_group, bank, weak)
    else:
        selected_groups = read_expected_selection(args.expected_selection)
    selected_keys = {
        (str(group["family_id"]), int(group["question_index"]))
        for group in selected_groups
    }
    if len(selected_keys) != 100:
        raise ValueError("selection must contain 100 unique family/question groups")
    if len({family_id for family_id, _ in selected_keys}) != 100:
        raise ValueError("selection must contain 100 distinct train families")
    missing_groups = sorted(key for key in selected_keys if key not in by_group)
    if missing_groups:
        raise ValueError(f"selected groups are absent from input manifest: {missing_groups}")
    selected = [
        row for key in selected_keys for row in by_group[key]
    ]
    selected.sort(
        key=lambda row: (
            str(row["scene_family_id"]),
            int(row["question_index"]),
            VARIANTS.index(str(row["variant_id"])),
        )
    )
    if len(selected) != 300 or len({str(row["id"]) for row in selected}) != 300:
        raise AssertionError(f"expected 300 unique records, got {len(selected)}")
    check_complete_groups(selected)
    if {str(row["split"]) for row in selected} != {"train"}:
        raise ValueError("pilot train manifest contains a non-train row")
    if len({str(row["scene_family_id"]) for row in selected}) != 100:
        raise ValueError("pilot train manifest must contain 100 scene families")
    if len({str(row["scene_family_id"]) for row in val_rows}) != 1:
        raise ValueError("pilot validation manifest must contain one family")
    if len(val_rows) != 48:
        raise ValueError("pilot validation family must contain 48 records")
    check_complete_groups(val_rows)
    if set(str(row["scene_family_id"]) for row in selected) & set(
        str(row["scene_family_id"]) for row in val_rows
    ):
        raise ValueError("train/validation scene families overlap")

    train_quality = source_quality(selected, bank, weak, reject_weak=False)
    val_quality = source_quality(val_rows, bank, weak, reject_weak=True)
    relation_counts = Counter(str(row["relation"]) for row in selected)
    no_evidence = sum(bool(row["no_evidence"]) for row in selected)
    if no_evidence < 60 or no_evidence > 120:
        raise ValueError(f"unexpected pilot no-evidence count: {no_evidence}")
    selected_group_rows = [serializable_group(group) for group in selected_groups]
    selected_group_rows.sort(
        key=lambda group: (
            str(group["family_id"]),
            int(group["question_index"]),
        )
    )

    write_jsonl(args.output_train, selected, args.overwrite)
    write_jsonl(args.output_val, val_rows, args.overwrite)
    if args.family_index_template is not None:
        assert args.family_index_output is not None
        write_family_index_list(
            template_path=args.family_index_template,
            output_path=args.family_index_output,
            selected_groups=selected_group_rows,
            overwrite=args.overwrite,
        )
    receipt = {
        "format": "qces_v5_clean_diverse_pilot300_v2",
        "purpose": "small-scale scene-disjoint pilot; not paper or test evidence",
        "train_manifest": identity(args.output_train),
        "validation_manifest": identity(args.output_val),
        "source_train_manifest": identity(args.train_manifest),
        "source_validation_manifest": identity(args.val_manifest),
        "selection": {
            "records_train ↑": len(selected),
            "records_validation ↑": len(val_rows),
            "train_family_count ↑": 100,
            "validation_family": str(val_rows[0]["scene_family_id"]),
            "complete_train_question_groups ↑": len(selected) // 3,
            "complete_validation_question_groups ↑": len(val_rows) // 3,
            "records_by_relation": dict(sorted(relation_counts.items())),
            "answerable_train_records ↑": len(selected) - no_evidence,
            "no_evidence_train_records ↑": no_evidence,
            "variants": list(VARIANTS),
            "selected_groups": selected_group_rows,
        },
        "source_quality": {
            "train": train_quality,
            "validation": val_quality,
            "curator_boundary": (
                "BEATs crop-bank probabilities rank the candidate sources; "
                "they are an engineering filter, not independent evaluation."
            ),
        },
        "authorization": {
            "paper_metric_allowed": False,
            "test_claim_allowed": False,
            "full_dataset_readiness_implied": False,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
