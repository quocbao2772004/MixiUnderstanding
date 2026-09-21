#!/usr/bin/env python3
"""Reference-free word voting over multiple enhanced ASR views."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable


MODES = ("audiosep_beta_0", "audiosep_oa_beta_0_25", "frcrn_beta_0", "frcrn_oa_beta_0_25", "frcrn_oa_beta_0_5")
EPSILON = "<eps>"


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _align(left: list[str], right: list[str]) -> tuple[list[tuple[str, int | None, int | None]], int]:
    table = [list(range(len(right) + 1))]
    for row_index, source in enumerate(left, 1):
        row = [row_index]
        for column, target in enumerate(right, 1):
            row.append(min(row[-1] + 1, table[-1][column] + 1, table[-1][column - 1] + (source != target)))
        table.append(row)
    row_index, column = len(left), len(right)
    operations: list[tuple[str, int | None, int | None]] = []
    while row_index or column:
        if row_index and column and table[row_index][column] == table[row_index - 1][column - 1] + (left[row_index - 1] != right[column - 1]):
            operations.append(("pair", row_index - 1, column - 1)); row_index -= 1; column -= 1
        elif row_index and table[row_index][column] == table[row_index - 1][column] + 1:
            operations.append(("delete", row_index - 1, None)); row_index -= 1
        else:
            operations.append(("insert", None, column - 1)); column -= 1
    return list(reversed(operations)), table[-1][-1]


def _distance(left: str, right: str) -> int:
    return _align(_words(left), _words(right))[1]


def _vote_word(counter: collections.Counter[str]) -> str:
    return max(counter, key=lambda word: (counter[word], word != EPSILON, word))


def rover(hypotheses: list[str]) -> str:
    sequences = [_words(value) for value in hypotheses]
    anchor = min(
        range(len(sequences)),
        key=lambda index: (sum(_align(sequences[index], other)[1] for other in sequences), index),
    )
    order = [anchor] + [index for index in range(len(sequences)) if index != anchor]
    slots = [collections.Counter({word: 1}) for word in sequences[anchor]]
    seen = 1
    for sequence_index in order[1:]:
        consensus = [_vote_word(slot) for slot in slots]
        operations, _ = _align(consensus, sequences[sequence_index])
        updated: list[collections.Counter[str]] = []
        for operation, left_index, right_index in operations:
            if operation == "pair":
                slots[int(left_index)][sequences[sequence_index][int(right_index)]] += 1
                updated.append(slots[int(left_index)])
            elif operation == "delete":
                slots[int(left_index)][EPSILON] += 1
                updated.append(slots[int(left_index)])
            else:
                updated.append(collections.Counter({EPSILON: seen, sequences[sequence_index][int(right_index)]: 1}))
        slots = updated
        seen += 1
    return " ".join(word for word in (_vote_word(slot) for slot in slots) if word != EPSILON)


def _load(path: Path) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    output: dict[tuple[str, str], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[(row["split"], row["scene_id"])][row["mode"]] = row
    return output


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edits = sum(row["edit_distance"] for row in rows)
    words = sum(row["reference_words"] for row in rows)
    return {
        "utterances": len(rows), "corpus_wer_↓": edits / max(words, 1),
        "accuracy_at_wer_0.25_↑": sum(row["wer_↓"] <= 0.25 for row in rows) / max(len(rows), 1),
        "total_edits": edits, "reference_words": words,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audiosep-items", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_audiosep_oa_v1/asr_items.jsonl")
    parser.add_argument("--frcrn-items", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_frcrn_oa_v1/asr_items.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_rover_v1")
    args = parser.parse_args()
    audiosep, frcrn = _load(args.audiosep_items.resolve()), _load(args.frcrn_items.resolve())
    rows: list[dict[str, Any]] = []
    for key in sorted(set(audiosep) & set(frcrn)):
        candidates = {**audiosep[key], **frcrn[key]}
        missing = [mode for mode in MODES if mode not in candidates]
        if missing:
            raise RuntimeError(f"{key}: missing {missing}")
        reference = candidates[MODES[0]]["reference"]
        hypothesis = rover([candidates[mode]["hypothesis"] for mode in MODES])
        edits, reference_words = _distance(reference, hypothesis), max(len(_words(reference)), 1)
        rows.append({
            "split": key[0], "scene_id": key[1], "reference": reference, "hypothesis": hypothesis,
            "edit_distance": edits, "reference_words": reference_words, "wer_↓": edits / reference_words,
            "candidate_modes": list(MODES),
            "candidate_hypotheses": {mode: candidates[mode]["hypothesis"] for mode in MODES},
        })
    summary = {split: _summary([row for row in rows if row["split"] == split]) for split in ("val", "test")}
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    item_path = output / "items.jsonl"
    _atomic_text(item_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt = {
        "format": "qces_vietnamese_transcript_rover_receipt_v1", "complete": True,
        "method": "reference-free progressive word confusion-network voting; medoid anchor",
        "candidate_modes_fixed": list(MODES), "ground_truth_used_for_fusion": False,
        "summary": summary, "items": _portable(item_path),
        "items_sha256": hashlib.sha256(item_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
