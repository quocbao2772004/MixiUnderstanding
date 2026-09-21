#!/usr/bin/env python3
"""Evaluate deterministic temporal QA over predicted event graphs.

Questions are generated only from the gold development graph.  The evaluated
system receives predicted event labels and intervals, never gold labels or QA
answers.  Four modes isolate the semantic and localization bottlenecks:
pooled-R1, learned semantics, oracle labels on predicted intervals, and
predicted labels on oracle intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mixi_understanding.qces.relational_event_slots_v1 import interval_iou_matrix
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _device,
    _sha256_file,
)
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import (
    NONE_LABEL,
    ResidualSlotSemanticHead,
)
from mixi_understanding.scripts.train_qces_v5_slot_count_head_v1 import (
    CountHeadConfig,
    SlotCountHead,
)


FORMAT = "qces_v5_event_graph_qa_evaluation_v1"
FRAMES = 250


@dataclass(frozen=True)
class Node:
    label_id: int
    onset: float
    offset: float
    score: float
    slot_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--count-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _mask(nodes: Sequence[Node], *, verification: str | None = None) -> torch.Tensor:
    value = torch.zeros(FRAMES, dtype=torch.bool)
    for node in nodes:
        start = max(0, min(FRAMES - 1, int(node.onset * FRAMES)))
        end = max(start + 1, min(FRAMES, int(torch.ceil(torch.tensor(node.offset * FRAMES)).item())))
        value[start:end] = True
    if nodes and verification == "after":
        start = max(0, min(FRAMES - 1, int(nodes[0].offset * FRAMES)))
        value[start:] = True
    elif nodes and verification == "before":
        end = max(1, min(FRAMES, int(torch.ceil(torch.tensor(nodes[0].onset * FRAMES)).item())))
        value[:end] = True
    return value


def _mask_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    union = int((left | right).sum())
    return float((left & right).sum()) / max(union, 1)


def _ordinal(nodes: Sequence[Node], index: int) -> int:
    label = nodes[index].label_id
    return 1 + sum(nodes[position].label_id == label for position in range(index))


def _questions(gold: Sequence[Node]) -> list[dict[str, Any]]:
    if not gold:
        return []
    result: list[dict[str, Any]] = [
        {"operation": "first_event", "answer": gold[0].label_id, "gold_nodes": [0]},
        {"operation": "last_event", "answer": gold[-1].label_id, "gold_nodes": [len(gold) - 1]},
        {
            "operation": "list_events",
            "answer": [node.label_id for node in gold],
            "gold_nodes": list(range(len(gold))),
        },
    ]
    for index, node in enumerate(gold):
        ordinal = _ordinal(gold, index)
        result.append(
            {
                "operation": "before",
                "anchor_label": node.label_id,
                "anchor_ordinal": ordinal,
                "answer": None if index == 0 else gold[index - 1].label_id,
                "gold_nodes": [index] if index == 0 else [index - 1, index],
                "no_evidence_reason": "no_event_before_anchor" if index == 0 else None,
            }
        )
        result.append(
            {
                "operation": "after",
                "anchor_label": node.label_id,
                "anchor_ordinal": ordinal,
                "answer": None if index == len(gold) - 1 else gold[index + 1].label_id,
                "gold_nodes": [index] if index == len(gold) - 1 else [index, index + 1],
                "no_evidence_reason": "no_event_after_anchor" if index == len(gold) - 1 else None,
            }
        )
    for left in range(len(gold) - 2):
        right = left + 2
        result.append(
            {
                "operation": "between",
                "left_label": gold[left].label_id,
                "left_ordinal": _ordinal(gold, left),
                "right_label": gold[right].label_id,
                "right_ordinal": _ordinal(gold, right),
                "answer": [gold[left + 1].label_id],
                "gold_nodes": [left, left + 1, right],
            }
        )
    return result


def _nth(nodes: Sequence[Node], label_id: int, ordinal: int) -> Node | None:
    candidates = [node for node in nodes if node.label_id == label_id]
    return candidates[ordinal - 1] if 0 < ordinal <= len(candidates) else None


def _execute(nodes: Sequence[Node], question: Mapping[str, Any]) -> tuple[Any, list[Node], str | None]:
    ordered = sorted(nodes, key=lambda node: (node.onset, node.offset, node.slot_index))
    operation = str(question["operation"])
    if operation == "first_event":
        return (ordered[0].label_id, [ordered[0]], None) if ordered else (None, [], "empty_graph")
    if operation == "last_event":
        return (ordered[-1].label_id, [ordered[-1]], None) if ordered else (None, [], "empty_graph")
    if operation == "list_events":
        return [node.label_id for node in ordered], list(ordered), None
    if operation in {"before", "after"}:
        anchor = _nth(
            ordered, int(question["anchor_label"]), int(question["anchor_ordinal"])
        )
        if anchor is None:
            return None, [], "anchor_missing"
        anchor_position = ordered.index(anchor)
        answer_position = anchor_position - 1 if operation == "before" else anchor_position + 1
        if answer_position < 0 or answer_position >= len(ordered):
            return None, [anchor], "no_valid_neighbor"
        answer = ordered[answer_position]
        return answer.label_id, [anchor, answer], None
    if operation == "between":
        left = _nth(ordered, int(question["left_label"]), int(question["left_ordinal"]))
        right = _nth(ordered, int(question["right_label"]), int(question["right_ordinal"]))
        if left is None or right is None:
            return None, [], "anchor_missing"
        lower, upper = sorted((left.onset, right.onset))
        between = [node for node in ordered if lower < node.onset < upper]
        return [node.label_id for node in between], [left, *between, right], None
    raise ValueError(f"unsupported operation: {operation}")


def _gold_nodes(cache: Mapping[str, Any], index: int) -> list[Node]:
    intervals = cache["gold_intervals"][index].float()
    labels = cache["gold_labels"][index].long()
    nodes = [
        Node(int(labels[pos]), float(intervals[pos, 0]), float(intervals[pos, 1]), 1.0, pos)
        for pos in range(len(labels))
    ]
    return sorted(nodes, key=lambda node: (node.onset, node.offset, node.slot_index))


def _predicted_nodes(
    cache: Mapping[str, Any],
    index: int,
    scores: torch.Tensor,
    objectness_threshold: float,
    selected_count: int | None = None,
) -> list[Node]:
    intervals = cache["intervals"][index].float()
    confidence = cache["objectness"][index].float()
    result: list[Node] = []
    if selected_count is None:
        selected_slots = torch.where(confidence >= objectness_threshold)[0].tolist()
    else:
        count = max(1, min(int(intervals.shape[0]), int(selected_count)))
        selected_slots = confidence.argsort(descending=True)[:count].tolist()
    for slot in selected_slots:
        label = int(scores[index, slot].argmax())
        if label == NONE_LABEL:
            continue
        result.append(
            Node(
                label,
                float(intervals[slot, 0]),
                float(intervals[slot, 1]),
                float(scores[index, slot, label]),
                slot,
            )
        )
    return sorted(result, key=lambda node: (node.onset, node.offset, node.slot_index))


def _oracle_labels_on_predicted_intervals(predicted: Sequence[Node], gold: Sequence[Node]) -> list[Node]:
    if not predicted or not gold:
        return []
    left = torch.tensor([[node.onset, node.offset] for node in predicted])
    right = torch.tensor([[node.onset, node.offset] for node in gold])
    iou = interval_iou_matrix(left, right)
    rows, columns = linear_sum_assignment((1.0 - iou).numpy())
    result: list[Node] = []
    for row, column in zip(rows, columns):
        if float(iou[row, column]) >= 0.5:
            source = predicted[row]
            result.append(
                Node(gold[column].label_id, source.onset, source.offset, 1.0, source.slot_index)
            )
    return sorted(result, key=lambda node: (node.onset, node.offset, node.slot_index))


def _predicted_labels_on_oracle_intervals(predicted: Sequence[Node], gold: Sequence[Node]) -> list[Node]:
    if not predicted or not gold:
        return []
    left = torch.tensor([[node.onset, node.offset] for node in predicted])
    right = torch.tensor([[node.onset, node.offset] for node in gold])
    iou = interval_iou_matrix(left, right)
    result: list[Node] = []
    for gold_index, target in enumerate(gold):
        slot_index = int(iou[:, gold_index].argmax())
        source = predicted[slot_index]
        result.append(Node(source.label_id, target.onset, target.offset, source.score, gold_index))
    return sorted(result, key=lambda node: (node.onset, node.offset, node.slot_index))


def _answer_equal(predicted: Any, gold: Any) -> bool:
    return predicted == gold


def _evaluate_mode(
    cache: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    objectness_threshold: float,
    mode: str,
    predicted_counts: torch.Tensor | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for scene_index, scene_id in enumerate(cache["scene_id"]):
        gold = _gold_nodes(cache, scene_index)
        predicted = _predicted_nodes(
            cache,
            scene_index,
            scores,
            objectness_threshold,
            None if predicted_counts is None else int(predicted_counts[scene_index]),
        )
        if mode.startswith("oracle_labels_"):
            graph = _oracle_labels_on_predicted_intervals(predicted, gold)
        elif mode == "predicted_labels_oracle_intervals":
            graph = _predicted_labels_on_oracle_intervals(predicted, gold)
        elif mode == "oracle_graph":
            graph = gold
        else:
            graph = predicted
        for question_index, question in enumerate(_questions(gold)):
            predicted_answer, evidence_nodes, failure = _execute(graph, question)
            gold_evidence_nodes = [gold[position] for position in question["gold_nodes"]]
            no_evidence = question["answer"] is None
            verification = question["operation"] if no_evidence else None
            evidence_iou = _mask_iou(
                _mask(evidence_nodes, verification=verification),
                _mask(gold_evidence_nodes, verification=verification),
            )
            answer_correct = _answer_equal(predicted_answer, question["answer"])
            verified_no_evidence = bool(
                no_evidence and answer_correct and failure != "anchor_missing" and evidence_iou >= 0.5
            )
            rows.append(
                {
                    "mode": mode,
                    "scene_id": str(scene_id),
                    "question_id": f"{scene_id}:{question_index:03d}",
                    "operation": question["operation"],
                    "answerable": not no_evidence,
                    "gold_answer": question["answer"],
                    "predicted_answer": predicted_answer,
                    "answer_correct": answer_correct,
                    "evidence_iou": evidence_iou,
                    "joint_answer_and_evidence_iou50": bool(answer_correct and evidence_iou >= 0.5),
                    "verified_no_evidence": verified_no_evidence,
                    "failure": failure,
                    "gold_graph_size": len(gold),
                    "predicted_graph_size": len(graph),
                }
            )

    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(items)
        answerable = [row for row in items if row["answerable"]]
        no_evidence = [row for row in items if not row["answerable"]]
        return {
            "questions": count,
            "answer_accuracy_↑": sum(row["answer_correct"] for row in items) / max(count, 1),
            "answerable_accuracy_↑": sum(row["answer_correct"] for row in answerable) / max(len(answerable), 1),
            "no_evidence_answer_accuracy_↑": sum(row["answer_correct"] for row in no_evidence) / max(len(no_evidence), 1),
            "verified_no_evidence_accuracy_↑": sum(row["verified_no_evidence"] for row in no_evidence) / max(len(no_evidence), 1),
            "mean_evidence_iou_↑": sum(float(row["evidence_iou"]) for row in items) / max(count, 1),
            "joint_answer_and_evidence_iou50_↑": sum(row["joint_answer_and_evidence_iou50"] for row in items) / max(count, 1),
        }

    by_operation = {
        operation: summarize([row for row in rows if row["operation"] == operation])
        for operation in sorted({str(row["operation"]) for row in rows})
    }
    return summarize(rows) | {"by_operation": by_operation}, rows


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cache_path = args.cache.resolve()
    semantic_path = args.semantic_checkpoint.resolve()
    count_path = args.count_checkpoint.resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    checkpoint = torch.load(semantic_path, map_location="cpu", weights_only=True)
    labels = list(checkpoint["labels"])
    if len(labels) != NONE_LABEL:
        raise ValueError(f"expected {NONE_LABEL} labels")
    device = _device(args.device)
    model = ResidualSlotSemanticHead(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        NONE_LABEL,
        float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    values = cache["slot_input"]
    learned_parts: list[torch.Tensor] = []
    for start in range(0, int(values.shape[0]), args.batch_size):
        learned_parts.append(model(values[start : start + args.batch_size].to(device)).cpu())
    learned = torch.cat(learned_parts)
    pooled_r1 = F.pad(values[..., -NONE_LABEL:].float(), (0, 1), value=-1e9)
    objectness_threshold = float(checkpoint["objectness_threshold"])

    count_checkpoint = torch.load(count_path, map_location="cpu", weights_only=True)
    count_model = SlotCountHead(CountHeadConfig(**count_checkpoint["config"])).to(device)
    count_model.load_state_dict(count_checkpoint["model_state_dict"], strict=True)
    count_model.eval()
    count_parts: list[torch.Tensor] = []
    count_embedding_dim = int(count_checkpoint["config"]["slot_embedding_dim"])
    for start in range(0, int(values.shape[0]), args.batch_size):
        stop = start + args.batch_size
        count_logits = count_model(
            values[start:stop, :, :count_embedding_dim].float().to(device),
            cache["objectness"][start:stop].float().to(device),
            cache["intervals"][start:stop].float().to(device),
        )
        count_parts.append(count_logits.argmax(dim=-1).clamp(1, 8).cpu())
    predicted_counts = torch.cat(count_parts)

    mode_scores = {
        "pooled_r1": (pooled_r1, None),
        "learned_semantic": (learned, None),
        "learned_semantic_count_head": (learned, predicted_counts),
        "oracle_labels_predicted_intervals": (learned, None),
        "oracle_labels_count_head_intervals": (learned, predicted_counts),
        "predicted_labels_oracle_intervals": (learned, None),
        "oracle_graph": (learned, None),
    }
    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for mode, (scores, counts) in mode_scores.items():
        summary, rows = _evaluate_mode(
            cache,
            scores,
            objectness_threshold=objectness_threshold,
            mode=mode,
            predicted_counts=counts,
        )
        summaries[mode] = summary
        all_rows.extend(rows)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split_name,
        "paper_eligible": False,
        "question_source": "deterministic_generation_from_gold_dev_event_graph",
        "gold_labels_or_answers_used_by_deployable_modes": False,
        "evidence_definition": "anchor_plus_answer; no-evidence uses anchor_plus_verification_region",
        "classes": len(labels),
        "scenes": len(cache["scene_id"]),
        "cache": str(cache_path),
        "cache_sha256": _sha256_file(cache_path),
        "semantic_checkpoint": str(semantic_path),
        "semantic_checkpoint_sha256": _sha256_file(semantic_path),
        "count_checkpoint": str(count_path),
        "count_checkpoint_sha256": _sha256_file(count_path),
        "objectness_threshold": objectness_threshold,
        "metrics": summaries,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(receipt, args.output.resolve())
    args.rows_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.rows_output.resolve().open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
