#!/usr/bin/env python3
"""Run one open-event QA query with the v2 inventory-aware reranker."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.evaluate_qces_open_event_v2 import (  # noqa: E402
    execute_v2,
    split_answer_context,
)
from mixi_understanding.scripts.infer_qces_open_event_qa import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DEFAULT_PROPOSAL_HEAD,
    EmbeddingTextParser,
    QwenTextParser,
    event_payload,
    load_inventory,
    parse_open_question,
    read_jsonl,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--scene-id")
    parser.add_argument("--record-id")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument("--proposal-head", type=Path, default=Path(DEFAULT_PROPOSAL_HEAD))
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--max-list-events", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--parser",
        choices=("rule", "embedding", "qwen"),
        default="embedding",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def resolve_scene_id(args: argparse.Namespace) -> str:
    if args.scene_id:
        return args.scene_id
    if not args.record_id or not args.manifest:
        raise SystemExit("provide --scene-id or both --record-id and --manifest")
    for row in read_jsonl(args.manifest):
        if row.get("id") == args.record_id:
            return str(row["scene_id"])
    raise SystemExit(f"record not found in manifest: {args.record_id}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scene_id = resolve_scene_id(args)
    labels, inventory = load_inventory(
        cache_path=args.cache.resolve(),
        proposal_head_path=args.proposal_head.resolve(),
        scene_id=scene_id,
        threshold=args.threshold,
        max_events=args.max_events,
        device_text=args.device,
    )
    llm = QwenTextParser(args.model, max_new_tokens=120) if args.parser == "qwen" else None
    embedding = (
        EmbeddingTextParser(args.embedding_model, device_text=args.device)
        if args.parser == "embedding"
        else None
    )
    program, raw, raw_json = parse_open_question(
        args.question,
        labels,
        llm,
        embedding_parser=embedding,
    )
    answer = execute_v2(program, inventory)
    answer_events, context_events = split_answer_context(answer)
    payload: dict[str, Any] = {
        "format": "qces_open_event_v2_infer",
        "scene_id": scene_id,
        "question": args.question,
        "parser": args.parser,
        "threshold": args.threshold,
        "available_label_count": len(labels),
        "inventory_event_count": len(inventory),
        "program": asdict(program),
        "parser_raw": raw,
        "parser_json": raw_json,
        "answer": answer.answer,
        "no_evidence": answer.no_evidence,
        "answer_reason": answer.reason,
        "answer_events": answer_events,
        "context_events": context_events,
        "evidence": [event_payload(event) for event in answer.evidence],
        "inventory_preview": [
            event_payload(event) for event in inventory[: args.max_list_events]
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
