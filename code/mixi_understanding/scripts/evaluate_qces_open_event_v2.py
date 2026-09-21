#!/usr/bin/env python3
"""Evaluate an inventory-aware open-event QA reranker.

This is a sidecar experiment for the Streamlit open-event demo.  It uses the
existing QCES-v6 separator-as-detector cache and proposal head, then evaluates
whether relation questions can be answered from the predicted event inventory.

The default mode uses oracle programs generated from the manifest.  That
intentionally removes parser errors so the measured bottleneck is:

    predicted inventory + answer-event reranker

If this fails on the current ~30-label setting, expanding to a 200-label prompt
bank will mainly increase false positives.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.qces.event_proposals import EventProposal
from mixi_understanding.scripts.infer_qces_open_event_qa import (
    OpenAnswer,
    OpenProgram,
    event_payload,
    load_inventory,
    parse_open_question,
    QwenTextParser,
    EmbeddingTextParser,
)


DEFAULT_CACHE = PROJECT_ROOT / "outputs/qces_v6_caches/val_stem_cache.pt"
DEFAULT_PROPOSAL_HEAD = (
    PROJECT_ROOT / "outputs/qces_v6_proposal_head_iou_v2/train_val_v1/proposal_head.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_open_event_v2"


@dataclass(frozen=True)
class EvalItem:
    sample_id: str
    scene_id: str
    question: str
    program: OpenProgram
    expected_answer: str
    expected_no_evidence: bool
    family: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--proposal-head", type=Path, default=DEFAULT_PROPOSAL_HEAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-items-per-scene", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--program-source",
        choices=("oracle", "embedding", "qwen", "rule"),
        default="oracle",
    )
    parser.add_argument(
        "--embedding-model",
        default=(
            "/home/cuongpv/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
            "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        ),
    )
    parser.add_argument(
        "--qwen-model",
        default=(
            "/home/cuongpv/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/"
            "snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_scene_rows(manifest: Path) -> OrderedDict[str, dict[str, Any]]:
    scenes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in read_jsonl(manifest):
        scene_id = row["scene_id"]
        if scene_id not in scenes:
            scenes[scene_id] = row
    return scenes


def cache_manifest(cache_path: Path) -> Path:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    return Path(cache["manifest"]).expanduser().resolve()


def ordinal_word(index: int) -> str:
    words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
    }
    return words.get(index, str(index))


def label_text(label: str) -> str:
    return label.replace("_", " ")


def semantic_events(row: Mapping[str, Any]) -> list[EventProposal]:
    events: list[EventProposal] = []
    for event in row.get("events") or []:
        if event.get("event_kind", "semantic") != "semantic":
            continue
        events.append(
            EventProposal(
                label=str(event["label"]),
                onset_seconds=float(event["onset_seconds"]),
                offset_seconds=float(event["offset_seconds"]),
                confidence=1.0,
            )
        )
    return sorted(events, key=lambda item: (item.onset_seconds, item.offset_seconds, item.label))


def occurrence_index(events: Sequence[EventProposal], position: int) -> int:
    label = events[position].label
    return 1 + sum(1 for event in events[:position] if event.label == label)


def unique_labels_chrono(events: Sequence[EventProposal]) -> list[str]:
    result: list[str] = []
    for event in events:
        if event.label not in result:
            result.append(event.label)
    return result


def next_distinct_label(events: Sequence[EventProposal], position: int) -> str | None:
    anchor = events[position]
    for event in events[position + 1 :]:
        if event.label != anchor.label:
            return event.label
    return None


def previous_distinct_label(events: Sequence[EventProposal], position: int) -> str | None:
    anchor = events[position]
    for event in reversed(events[:position]):
        if event.label != anchor.label:
            return event.label
    return None


def build_eval_items(
    scene_rows: Mapping[str, Mapping[str, Any]],
    cache_labels: Mapping[str, Sequence[str]],
    *,
    max_items_per_scene: int,
) -> list[EvalItem]:
    items: list[EvalItem] = []
    for scene_id, row in scene_rows.items():
        events = semantic_events(row)
        if not events:
            continue
        labels_present = set(event.label for event in events)
        available_labels = list(cache_labels.get(scene_id) or [])
        absent_labels = [label for label in available_labels if label not in labels_present]
        local: list[EvalItem] = []

        first = events[0]
        last = max(events, key=lambda item: item.onset_seconds)
        longest = max(events, key=lambda item: (item.duration_seconds, item.confidence))
        local.append(
            EvalItem(
                f"{scene_id}:first",
                scene_id,
                "What is the first sound in the audio?",
                OpenProgram("first_event"),
                first.label,
                False,
                "first_event",
            )
        )
        local.append(
            EvalItem(
                f"{scene_id}:last",
                scene_id,
                "What is the last sound in the audio?",
                OpenProgram("last_event"),
                last.label,
                False,
                "last_event",
            )
        )
        local.append(
            EvalItem(
                f"{scene_id}:longest",
                scene_id,
                "Which sound lasts the longest?",
                OpenProgram("longest_event"),
                longest.label,
                False,
                "longest_event",
            )
        )
        for label in unique_labels_chrono(events)[:4]:
            count = sum(1 for event in events if event.label == label)
            text = label_text(label)
            local.append(
                EvalItem(
                    f"{scene_id}:exists:{label}",
                    scene_id,
                    f"Is there a {text} sound?",
                    OpenProgram("exists", target_label=label),
                    "yes",
                    False,
                    "exists_pos",
                )
            )
            local.append(
                EvalItem(
                    f"{scene_id}:count:{label}",
                    scene_id,
                    f"How many times does {text} occur?",
                    OpenProgram("count", target_label=label),
                    str(count),
                    False,
                    "count",
                )
            )
        for label in absent_labels[:3]:
            text = label_text(label)
            local.append(
                EvalItem(
                    f"{scene_id}:absent:{label}",
                    scene_id,
                    f"Is there a {text} sound?",
                    OpenProgram("exists", target_label=label),
                    "no",
                    True,
                    "exists_neg",
                )
            )
        for pos, event in enumerate(events):
            ordinal = occurrence_index(events, pos)
            text = label_text(event.label)
            answer = next_distinct_label(events, pos)
            if answer is not None:
                local.append(
                    EvalItem(
                        f"{scene_id}:after:{pos}",
                        scene_id,
                        f"What sound happens after the {ordinal_word(ordinal)} {text} sound?",
                        OpenProgram(
                            "after",
                            anchor_label=event.label,
                            anchor_ordinal=ordinal,
                        ),
                        answer,
                        False,
                        "after",
                    )
                )
            answer = previous_distinct_label(events, pos)
            if answer is not None:
                local.append(
                    EvalItem(
                        f"{scene_id}:before:{pos}",
                        scene_id,
                        f"What sound happens before the {ordinal_word(ordinal)} {text} sound?",
                        OpenProgram(
                            "before",
                            anchor_label=event.label,
                            anchor_ordinal=ordinal,
                        ),
                        answer,
                        False,
                        "before",
                    )
                )
        for left_idx in range(0, max(0, len(events) - 2)):
            middle = events[left_idx + 1]
            right = events[left_idx + 2]
            left = events[left_idx]
            if len({left.label, middle.label, right.label}) < 3:
                continue
            ordinal = occurrence_index(events, left_idx)
            local.append(
                EvalItem(
                    f"{scene_id}:between:{left_idx}",
                    scene_id,
                    (
                        f"What sound is between the {ordinal_word(ordinal)} "
                        f"{label_text(left.label)} sound and {label_text(right.label)}?"
                    ),
                    OpenProgram(
                        "between",
                        anchor_label=left.label,
                        target_label=right.label,
                        anchor_ordinal=ordinal,
                        candidate_labels=(left.label, right.label),
                    ),
                    middle.label,
                    False,
                    "between",
                )
            )
        items.extend(local[:max_items_per_scene] if max_items_per_scene else local)
    return items


def occurrences(events: Sequence[EventProposal], label: str) -> list[EventProposal]:
    return sorted(
        (event for event in events if event.label == label),
        key=lambda event: (event.onset_seconds, event.offset_seconds, -event.confidence),
    )


def temporal_overlap(a: EventProposal, b: EventProposal) -> float:
    inter = max(0.0, min(a.offset_seconds, b.offset_seconds) - max(a.onset_seconds, b.onset_seconds))
    union = max(a.offset_seconds, b.offset_seconds) - min(a.onset_seconds, b.onset_seconds)
    return inter / max(union, 1e-6)


def nearest_legacy(program: OpenProgram, events: Sequence[EventProposal]) -> OpenAnswer:
    ordered = sorted(events, key=lambda item: (item.onset_seconds, item.offset_seconds, -item.confidence))
    if not program.ok or not ordered:
        return OpenAnswer("no_evidence", True, (), "legacy_empty_or_bad_program")
    if program.operation == "first_event":
        return OpenAnswer(ordered[0].label, False, (ordered[0],), "legacy_first")
    if program.operation == "last_event":
        event = max(ordered, key=lambda item: item.onset_seconds)
        return OpenAnswer(event.label, False, (event,), "legacy_last")
    if program.operation == "longest_event":
        event = max(ordered, key=lambda item: item.duration_seconds)
        return OpenAnswer(event.label, False, (event,), "legacy_longest")
    if program.operation == "exists":
        occ = occurrences(ordered, str(program.target_label))
        return OpenAnswer("yes" if occ else "no", not bool(occ), tuple(occ), "legacy_exists")
    if program.operation == "count":
        occ = occurrences(ordered, str(program.target_label))
        return OpenAnswer(str(len(occ)), not bool(occ), tuple(occ), "legacy_count")
    if program.operation in {"after", "before"}:
        anchors = occurrences(ordered, str(program.anchor_label))
        if len(anchors) < program.anchor_ordinal:
            return OpenAnswer("no_evidence", True, (), "legacy_anchor_missing")
        anchor = anchors[program.anchor_ordinal - 1]
        if program.operation == "after":
            candidates = [e for e in ordered if e is not anchor and e.onset_seconds > anchor.onset_seconds]
            if not candidates:
                return OpenAnswer("no_evidence", True, (anchor,), "legacy_after_missing")
            answer = min(candidates, key=lambda item: item.onset_seconds)
        else:
            candidates = [e for e in ordered if e is not anchor and e.onset_seconds < anchor.onset_seconds]
            if not candidates:
                return OpenAnswer("no_evidence", True, (anchor,), "legacy_before_missing")
            answer = max(candidates, key=lambda item: item.onset_seconds)
        return OpenAnswer(answer.label, False, tuple(sorted((anchor, answer), key=lambda x: x.onset_seconds)), "legacy_relation")
    if program.operation == "between":
        lefts = occurrences(ordered, str(program.anchor_label))
        if len(lefts) < program.anchor_ordinal:
            return OpenAnswer("no_evidence", True, (), "legacy_left_missing")
        left = lefts[program.anchor_ordinal - 1]
        rights = [e for e in occurrences(ordered, str(program.target_label)) if e.onset_seconds > left.onset_seconds]
        if not rights:
            return OpenAnswer("no_evidence", True, (left,), "legacy_right_missing")
        right = min(rights, key=lambda item: item.onset_seconds)
        candidates = [
            e
            for e in ordered
            if e.label not in {left.label, right.label}
            and e.onset_seconds > left.onset_seconds
            and e.onset_seconds < right.onset_seconds
        ]
        if not candidates:
            return OpenAnswer("no_evidence", True, (left, right), "legacy_between_missing")
        answer = max(candidates, key=lambda item: item.confidence)
        return OpenAnswer(answer.label, False, tuple(sorted((left, answer, right), key=lambda x: x.onset_seconds)), "legacy_between")
    return OpenAnswer("unsupported", True, (), "legacy_unsupported")


def score_after(anchor: EventProposal, candidate: EventProposal) -> float:
    delta_onset = candidate.onset_seconds - anchor.onset_seconds
    if delta_onset <= 0:
        return -1e9
    confidence = candidate.confidence
    # Relation questions are primarily about chronological proximity.  A high
    # global confidence should not let a later event skip over a low-energy but
    # temporally correct event, while heavy overlap is often a false-positive
    # alias of the anchor.
    overlap_penalty = 2.0 * temporal_overlap(anchor, candidate)
    temporal = -1.0 * delta_onset
    return confidence + temporal - overlap_penalty


def score_before(anchor: EventProposal, candidate: EventProposal) -> float:
    delta = anchor.onset_seconds - candidate.onset_seconds
    if delta <= 0:
        return -1e9
    confidence = candidate.confidence
    overlap_penalty = 2.0 * temporal_overlap(anchor, candidate)
    temporal = -1.0 * delta
    return confidence + temporal - overlap_penalty


def execute_v2(program: OpenProgram, events: Sequence[EventProposal]) -> OpenAnswer:
    ordered = sorted(events, key=lambda item: (item.onset_seconds, item.offset_seconds, -item.confidence))
    if not program.ok or not ordered:
        return OpenAnswer("no_evidence", True, (), "v2_empty_or_bad_program")
    if program.operation == "first_event":
        candidates = [event for event in ordered if event.confidence >= 0.30] or ordered
        event = min(candidates, key=lambda item: (item.onset_seconds, -item.confidence))
        return OpenAnswer(event.label, False, (event,), "v2_first_confident")
    if program.operation == "last_event":
        candidates = [event for event in ordered if event.confidence >= 0.30] or ordered
        event = max(candidates, key=lambda item: (item.onset_seconds, item.confidence))
        return OpenAnswer(event.label, False, (event,), "v2_last_confident")
    if program.operation == "longest_event":
        candidates = [event for event in ordered if event.confidence >= 0.30] or ordered
        event = max(candidates, key=lambda item: (item.duration_seconds * (0.5 + item.confidence), item.confidence))
        return OpenAnswer(event.label, False, (event,), "v2_longest_conf_weighted")
    if program.operation == "exists":
        occ = occurrences(ordered, str(program.target_label))
        good = [event for event in occ if event.confidence >= 0.30]
        return OpenAnswer("yes" if good else "no", not bool(good), tuple(good), "v2_exists_conf")
    if program.operation == "count":
        occ = [event for event in occurrences(ordered, str(program.target_label)) if event.confidence >= 0.30]
        return OpenAnswer(str(len(occ)), not bool(occ), tuple(occ), "v2_count_conf")
    if program.operation in {"after", "before"}:
        anchors = [event for event in occurrences(ordered, str(program.anchor_label)) if event.confidence >= 0.20]
        if len(anchors) < program.anchor_ordinal:
            return OpenAnswer("no_evidence", True, (), "v2_anchor_missing")
        anchor = anchors[program.anchor_ordinal - 1]
        if program.operation == "after":
            candidates = [
                event
                for event in ordered
                if event.label != anchor.label
                and event.onset_seconds > anchor.onset_seconds + 1e-6
                and event.confidence >= 0.20
            ]
            scorer = lambda event: score_after(anchor, event)
            missing_reason = "v2_after_missing"
        else:
            candidates = [
                event
                for event in ordered
                if event.label != anchor.label
                and event.onset_seconds < anchor.onset_seconds - 1e-6
                and event.confidence >= 0.20
            ]
            scorer = lambda event: score_before(anchor, event)
            missing_reason = "v2_before_missing"
        if not candidates:
            return OpenAnswer("no_evidence", True, (anchor,), missing_reason)
        answer = max(candidates, key=scorer)
        return OpenAnswer(
            answer.label,
            False,
            tuple(sorted((anchor, answer), key=lambda x: x.onset_seconds)),
            f"v2_{program.operation}_reranked",
        )
    if program.operation == "between":
        lefts = [event for event in occurrences(ordered, str(program.anchor_label)) if event.confidence >= 0.20]
        if len(lefts) < program.anchor_ordinal:
            return OpenAnswer("no_evidence", True, (), "v2_left_missing")
        left = lefts[program.anchor_ordinal - 1]
        rights = [
            event
            for event in occurrences(ordered, str(program.target_label))
            if event.onset_seconds > left.onset_seconds + 1e-6 and event.confidence >= 0.20
        ]
        if not rights:
            return OpenAnswer("no_evidence", True, (left,), "v2_right_missing")
        right = min(rights, key=lambda item: item.onset_seconds)
        candidates = [
            event
            for event in ordered
            if event.label not in {left.label, right.label}
            and event.onset_seconds > left.onset_seconds + 1e-6
            and event.onset_seconds < right.onset_seconds - 1e-6
            and event.confidence >= 0.20
        ]
        if not candidates:
            return OpenAnswer("no_evidence", True, tuple(sorted((left, right), key=lambda x: x.onset_seconds)), "v2_between_missing")
        center = 0.5 * (left.onset_seconds + right.onset_seconds)
        span = max(right.onset_seconds - left.onset_seconds, 1e-6)
        answer = max(
            candidates,
            key=lambda item: 2.4 * item.confidence - abs(item.onset_seconds - center) / span,
        )
        return OpenAnswer(
            answer.label,
            False,
            tuple(sorted((left, answer, right), key=lambda x: x.onset_seconds)),
            "v2_between_reranked",
        )
    if program.operation == "compare_first":
        firsts: list[EventProposal] = []
        for label in program.candidate_labels[:2]:
            occ = [event for event in occurrences(ordered, label) if event.confidence >= 0.20]
            if not occ:
                return OpenAnswer("no_evidence", True, tuple(firsts), f"v2_candidate_missing:{label}")
            firsts.append(occ[0])
        evidence = tuple(sorted(firsts, key=lambda item: item.onset_seconds))
        return OpenAnswer(evidence[0].label, False, evidence, "v2_compare_first")
    return OpenAnswer("unsupported", True, (), "v2_unsupported")


def split_answer_context(answer: OpenAnswer) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = {piece.strip() for piece in answer.answer.split(",") if piece.strip()}
    labels -= {"yes", "no", "no_evidence", "unsupported"}
    answer_events: list[dict[str, Any]] = []
    context_events: list[dict[str, Any]] = []
    for event in answer.evidence:
        payload = event_payload(event)
        if event.label in labels:
            answer_events.append(payload)
        else:
            context_events.append(payload)
    return answer_events, context_events


def accuracy(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    manifest = Path(cache["manifest"]).expanduser().resolve()
    scene_rows = load_scene_rows(manifest)
    cache_labels = {
        scene_id: tuple(entry["labels"])
        for scene_id, entry in cache["scenes"].items()
    }
    scene_ids = [scene_id for scene_id in cache["scenes"].keys() if scene_id in scene_rows]
    if args.max_scenes:
        scene_ids = scene_ids[: args.max_scenes]
    selected_rows = OrderedDict((scene_id, scene_rows[scene_id]) for scene_id in scene_ids)
    items = build_eval_items(
        selected_rows,
        cache_labels,
        max_items_per_scene=args.max_items_per_scene,
    )
    print(
        f"scenes={len(selected_rows)} items={len(items)} program_source={args.program_source}",
        flush=True,
    )

    embedding_parser = None
    qwen_parser = None
    if args.program_source == "embedding":
        embedding_parser = EmbeddingTextParser(args.embedding_model, device_text=args.device)
    elif args.program_source == "qwen":
        qwen_parser = QwenTextParser(args.qwen_model, max_new_tokens=120)

    inventory_by_scene: dict[str, tuple[str, ...] | tuple[EventProposal, ...]] = {}
    labels_by_scene: dict[str, tuple[str, ...]] = {}
    events_by_scene: dict[str, tuple[EventProposal, ...]] = {}
    for index, scene_id in enumerate(scene_ids, start=1):
        labels, inventory = load_inventory(
            cache_path=args.cache,
            proposal_head_path=args.proposal_head,
            scene_id=scene_id,
            threshold=args.threshold,
            max_events=args.max_events,
            device_text=args.device,
        )
        labels_by_scene[scene_id] = labels
        events_by_scene[scene_id] = inventory
        if index % 25 == 0 or index == len(scene_ids):
            print(f"decoded scene {index}/{len(scene_ids)}", flush=True)

    result_rows: list[dict[str, Any]] = []
    family_counter: Counter[str] = Counter()
    legacy_counter: Counter[str] = Counter()
    v2_counter: Counter[str] = Counter()
    for item in items:
        labels = labels_by_scene[item.scene_id]
        if args.program_source == "oracle":
            program = item.program
            parser_raw: str | None = None
            parser_json: Any = None
        elif args.program_source == "rule":
            program, parser_raw, parser_json = parse_open_question(item.question, labels, None)
        elif args.program_source == "embedding":
            program, parser_raw, parser_json = parse_open_question(
                item.question, labels, None, embedding_parser=embedding_parser
            )
        else:
            program, parser_raw, parser_json = parse_open_question(item.question, labels, qwen_parser)

        predicted_events = events_by_scene[item.scene_id]
        legacy = nearest_legacy(program, predicted_events)
        v2 = execute_v2(program, predicted_events)
        oracle = execute_v2(program, semantic_events(selected_rows[item.scene_id]))
        legacy_ok = legacy.answer == item.expected_answer and legacy.no_evidence == item.expected_no_evidence
        v2_ok = v2.answer == item.expected_answer and v2.no_evidence == item.expected_no_evidence
        oracle_ok = oracle.answer == item.expected_answer and oracle.no_evidence == item.expected_no_evidence
        family_counter[item.family] += 1
        legacy_counter[item.family] += int(legacy_ok)
        v2_counter[item.family] += int(v2_ok)
        answer_events, context_events = split_answer_context(v2)
        result_rows.append(
            {
                "sample_id": item.sample_id,
                "scene_id": item.scene_id,
                "family": item.family,
                "question": item.question,
                "expected_answer": item.expected_answer,
                "expected_no_evidence": item.expected_no_evidence,
                "program": asdict(program),
                "parser_raw": parser_raw,
                "parser_json": parser_json,
                "oracle_answer": oracle.answer,
                "oracle_ok": oracle_ok,
                "legacy_answer": legacy.answer,
                "legacy_no_evidence": legacy.no_evidence,
                "legacy_ok": legacy_ok,
                "legacy_reason": legacy.reason,
                "v2_answer": v2.answer,
                "v2_no_evidence": v2.no_evidence,
                "v2_ok": v2_ok,
                "v2_reason": v2.reason,
                "v2_answer_events": answer_events,
                "v2_context_events": context_events,
                "v2_evidence": [event_payload(event) for event in v2.evidence],
            }
        )

    family_rows = []
    for family, total in sorted(family_counter.items()):
        subset = [row for row in result_rows if row["family"] == family]
        family_rows.append(
            {
                "family": family,
                "n": total,
                "legacy_acc_↑": legacy_counter[family] / total,
                "v2_acc_↑": v2_counter[family] / total,
                "oracle_program_acc_↑": accuracy(subset, "oracle_ok"),
            }
        )

    summary = {
        "format": "qces_open_event_v2_eval",
        "cache": str(args.cache.resolve()),
        "proposal_head": str(args.proposal_head.resolve()),
        "manifest": str(manifest),
        "threshold": args.threshold,
        "max_events": args.max_events,
        "program_source": args.program_source,
        "scenes": len(selected_rows),
        "items": len(result_rows),
        "unique_cache_labels": len({label for labels in cache_labels.values() for label in labels}),
        "median_labels_per_scene": sorted(len(labels) for labels in cache_labels.values())[len(cache_labels) // 2],
        "legacy_acc_↑": accuracy(result_rows, "legacy_ok"),
        "v2_acc_↑": accuracy(result_rows, "v2_ok"),
        "oracle_program_acc_↑": accuracy(result_rows, "oracle_ok"),
        "families": family_rows,
    }
    result_path = output_dir / "predictions.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# QCES open-event v2 evaluation",
        "",
        f"- scenes: {summary['scenes']}",
        f"- items: {summary['items']}",
        f"- unique cache labels: {summary['unique_cache_labels']}",
        f"- median labels / scene: {summary['median_labels_per_scene']}",
        f"- threshold: {summary['threshold']}",
        f"- program source: {summary['program_source']}",
        "",
        "| method | accuracy ↑ |",
        "|---|---:|",
        f"| oracle inventory/program | {summary['oracle_program_acc_↑']:.3f} |",
        f"| legacy executor | {summary['legacy_acc_↑']:.3f} |",
        f"| open-event v2 reranker | {summary['v2_acc_↑']:.3f} |",
        "",
        "| family | n | legacy acc ↑ | v2 acc ↑ | oracle acc ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['n']} | {row['legacy_acc_↑']:.3f} | {row['v2_acc_↑']:.3f} | {row['oracle_program_acc_↑']:.3f} |"
        )
    lines.append("")
    lines.append("Worst v2 errors are in `predictions.jsonl` where `v2_ok=false`.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
