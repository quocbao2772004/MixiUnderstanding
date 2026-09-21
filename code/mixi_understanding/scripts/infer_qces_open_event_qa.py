#!/usr/bin/env python3
"""Open-ish event QA over a QCES full-scene event inventory.

This is a prototype sidecar for widening QCES beyond fixed 5-way questions.
It does not edit the current benchmark/evidence pipeline.

Flow:

    user question
        -> local Qwen text parser -> structured event program
        -> QCES proposal-head full inventory over all cached scene labels
        -> symbolic executor -> answer + evidence events

The answer is constrained by the current QCES label inventory.  It is not yet a
fully open-vocabulary AF3 replacement, but it removes the current 4-option
question limitation.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.qces.event_proposals import (
    EventProposal,
    ProposalHead,
    decode_proposals,
    features_from_cache,
)
from mixi_understanding.qces.stem_features import FrameGrid
from mixi_understanding.scripts.evaluate_qces_llm_question_parser import (
    DEFAULT_MODEL,
    QwenTextParser,
    extract_json,
    infer_anchor_from_question,
    normalize_label,
)


DEFAULT_CACHE = "outputs/qces_v6_caches/val_stem_cache.pt"
DEFAULT_PROPOSAL_HEAD = "outputs/qces_v6_proposal_head_iou_v2/train_val_v1/proposal_head.pt"
DEFAULT_EMBEDDING_MODEL = (
    "/home/cuongpv/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)


@dataclass(frozen=True)
class OpenProgram:
    operation: str
    target_label: str | None = None
    anchor_label: str | None = None
    anchor_ordinal: int = 1
    candidate_labels: tuple[str, ...] = ()
    ok: bool = True
    reason: str = "ok"


@dataclass(frozen=True)
class OpenAnswer:
    answer: str
    no_evidence: bool
    evidence: tuple[EventProposal, ...]
    reason: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--scene-id")
    parser.add_argument("--record-id")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument("--proposal-head", type=Path, default=Path(DEFAULT_PROPOSAL_HEAD))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--parser",
        choices=("qwen", "embedding", "rule"),
        default="qwen",
        help="Question parser. --no-llm is kept as a backward-compatible alias for --parser rule.",
    )
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--max-events", type=int, default=40)
    parser.add_argument("--max-list-events", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_scene_id(args: argparse.Namespace) -> str:
    if args.scene_id:
        return args.scene_id
    if not args.record_id or not args.manifest:
        raise SystemExit("provide --scene-id or both --record-id and --manifest")
    for row in read_jsonl(args.manifest):
        if row.get("id") == args.record_id:
            return str(row["scene_id"])
    raise SystemExit(f"record not found in manifest: {args.record_id}")


def load_inventory(
    *,
    cache_path: Path,
    proposal_head_path: Path,
    scene_id: str,
    threshold: float,
    max_events: int,
    device_text: str,
) -> tuple[tuple[str, ...], tuple[EventProposal, ...]]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if scene_id not in cache["scenes"]:
        raise SystemExit(f"scene_id not found in cache: {scene_id}")
    entry = cache["scenes"][scene_id]
    labels, features, _stems = features_from_cache(entry)
    ckpt = torch.load(proposal_head_path, map_location="cpu", weights_only=False)
    model = ProposalHead(
        channels=int(ckpt.get("channels", 96)),
        dropout=float(ckpt.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device(
        "cuda"
        if device_text == "auto" and torch.cuda.is_available()
        else "cpu" if device_text == "auto" else device_text
    )
    model.to(device).eval()
    with torch.inference_mode():
        logits, onset_logits, presence = model(features.to(device))
    activity = (torch.sigmoid(logits) * torch.sigmoid(presence)[:, None]).cpu()
    onset = torch.sigmoid(onset_logits).cpu()
    proposals = decode_proposals(
        labels,
        activity,
        FrameGrid(sample_rate=32_000),
        threshold=threshold,
        onset_activity=onset,
    )
    if max_events > 0 and len(proposals) > max_events:
        keep = sorted(proposals, key=lambda item: item.confidence, reverse=True)[:max_events]
        proposals = sorted(keep, key=lambda item: (item.onset_seconds, item.offset_seconds, item.label))
    else:
        proposals = sorted(proposals, key=lambda item: (item.onset_seconds, item.offset_seconds, item.label))
    return labels, tuple(proposals)


def open_parser_prompt(question: str, labels: Sequence[str]) -> str:
    return (
        "You are a strict parser for audio event questions. Return JSON only.\n\n"
        "Available operations:\n"
        "- list_events: ask what sounds/events are present.\n"
        "- exists: ask whether a target sound is present.\n"
        "- count: ask how many times a target sound occurs.\n"
        "- first_event: ask what event starts first.\n"
        "- last_event: ask what event starts last.\n"
        "- longest_event: ask what event lasts longest.\n"
        "- after: ask what starts after a named anchor occurrence.\n"
        "- before: ask what starts before a named anchor occurrence.\n"
        "- between: ask what sound(s) occur between two named sounds.\n"
        "- compare_first: ask which of two named sounds starts earlier.\n"
        "- unsupported: anything else.\n\n"
        "JSON schema:\n"
        "{"
        '"operation":"list_events|exists|count|first_event|last_event|longest_event|after|before|between|compare_first|unsupported",'
        '"target_label":"label or null",'
        '"anchor_label":"label or null",'
        '"anchor_ordinal":1,'
        '"candidate_labels":["label1","label2"]'
        "}\n\n"
        "Rules:\n"
        "- Use labels exactly from Available labels.\n"
        "- For exists/count, fill target_label.\n"
        "- For after/before, fill anchor_label and anchor_ordinal.\n"
        "- For between, fill anchor_label as the left boundary, target_label as the right boundary, and anchor_ordinal for the left boundary occurrence.\n"
        "- For compare_first, fill candidate_labels with two labels.\n"
        "- For list/first/last/longest, labels can be null/empty.\n\n"
        "Examples:\n"
        'Question: What sounds are in this audio?\n{"operation":"list_events","target_label":null,"anchor_label":null,"anchor_ordinal":1,"candidate_labels":[]}\n'
        'Question: Is there a Printer sound?\n{"operation":"exists","target_label":"Printer","anchor_label":null,"anchor_ordinal":1,"candidate_labels":[]}\n'
        'Question: How many times does Camera occur?\n{"operation":"count","target_label":"Camera","anchor_label":null,"anchor_ordinal":1,"candidate_labels":[]}\n'
        'Question: What starts right after the first Microwave_oven?\n{"operation":"after","target_label":null,"anchor_label":"Microwave_oven","anchor_ordinal":1,"candidate_labels":[]}\n'
        'Question: What sound is between the third Camera sound and Slam?\n{"operation":"between","target_label":"Slam","anchor_label":"Camera","anchor_ordinal":3,"candidate_labels":["Camera","Slam"]}\n'
        'Question: Between Camera and Printer, which starts earlier?\n{"operation":"compare_first","target_label":null,"anchor_label":null,"anchor_ordinal":1,"candidate_labels":["Camera","Printer"]}\n\n'
        f"Available labels: {list(labels)}\n"
        f"Question: {question}\n"
    )


def relation_cue(question: str) -> str | None:
    lowered = question.lower()
    if any(
        cue in lowered
        for cue in (
            "right after",
            "after the",
            "after it",
            "begins next",
            "starts next",
            "what starts after",
            "what begins after",
            "what sound after",
            "which sound after",
            "sound after",
            "happens after",
        )
    ):
        return "after"
    if any(
        cue in lowered
        for cue in (
            "right before",
            "just before",
            "before the",
            "before it",
            "preceding",
            "prior",
            "what sound before",
            "which sound before",
            "sound before",
            "happens before",
        )
    ):
        return "before"
    return None


def ordinal_from_question(question: str) -> int | None:
    words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    lowered = question.lower()
    for word, value in words.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    return None


def label_variants(label: str) -> tuple[str, ...]:
    raw = label.strip()
    spaced = re.sub(r"[_/]+", " ", raw)
    cleaned = re.sub(r"\([^)]*\)", "", spaced)
    variants = {
        raw,
        spaced,
        cleaned,
        re.sub(r"[^A-Za-z0-9]+", " ", raw).strip(),
        re.sub(r"[^A-Za-z0-9]+", " ", cleaned).strip(),
    }
    return tuple(sorted((item for item in variants if item), key=len, reverse=True))


def normalize_operation(raw: str) -> str:
    key = raw.strip().lower()
    aliases = {
        "list": "list_events",
        "events": "list_events",
        "presence": "exists",
        "present": "exists",
        "existence": "exists",
        "how_many": "count",
        "in_between": "between",
        "first": "first_event",
        "earliest": "first_event",
        "last": "last_event",
        "latest": "last_event",
        "longest": "longest_event",
        "compare": "compare_first",
        "which_first": "compare_first",
        "earlier_of_two": "compare_first",
    }
    key = aliases.get(key, key)
    allowed = {
        "list_events",
        "exists",
        "count",
        "first_event",
        "last_event",
        "longest_event",
        "after",
        "before",
        "between",
        "compare_first",
        "unsupported",
    }
    return key if key in allowed else "unsupported"


def normalize_program(obj: Mapping[str, Any] | None, labels: Sequence[str], question: str) -> OpenProgram:
    if obj is None:
        return rule_open_program(question, labels)
    operation = normalize_operation(str(obj.get("operation", "")))
    cue = relation_cue(question)
    if cue in {"after", "before"}:
        operation = cue
    target = normalize_label(obj.get("target_label"), labels)
    anchor = normalize_label(obj.get("anchor_label"), labels)
    candidates_raw = obj.get("candidate_labels") or []
    if isinstance(candidates_raw, str):
        candidates_raw = [candidates_raw]
    candidates = tuple(
        label
        for label in (normalize_label(item, labels) for item in candidates_raw)
        if label is not None
    )
    try:
        ordinal = int(obj.get("anchor_ordinal") or 1)
    except (TypeError, ValueError):
        ordinal = 1
    inferred_anchor, inferred_ordinal = infer_anchor_from_question(question, labels)
    mentioned = labels_mentioned(question, labels)
    if operation in {"after", "before"}:
        anchor = anchor or inferred_anchor or (mentioned[0] if mentioned else None)
        ordinal = inferred_ordinal or ordinal
        if anchor is None:
            return OpenProgram(operation, anchor_label=None, anchor_ordinal=ordinal, ok=False, reason="missing_anchor")
    if operation == "between":
        anchor = anchor or (candidates[0] if len(candidates) >= 1 else None) or (mentioned[0] if len(mentioned) >= 1 else None)
        target = target or (candidates[1] if len(candidates) >= 2 else None) or (mentioned[1] if len(mentioned) >= 2 else None)
        if anchor is not None and target == anchor:
            if len(candidates) >= 1 and candidates[0] != anchor:
                target = candidates[0]
            elif len(mentioned) >= 2:
                target = mentioned[1]
        if len(mentioned) >= 2 and (anchor not in mentioned or target not in mentioned):
            anchor = mentioned[0]
            target = mentioned[1]
        ordinal = ordinal_from_question(question) or inferred_ordinal or ordinal
        if anchor is None or target is None:
            return OpenProgram(
                operation,
                target_label=target,
                anchor_label=anchor,
                anchor_ordinal=ordinal,
                candidate_labels=candidates,
                ok=False,
                reason="missing_between_boundaries",
            )
    if operation in {"exists", "count"}:
        target = target or inferred_anchor or (mentioned[0] if mentioned else None)
        if target is None:
            return OpenProgram(operation, target_label=None, ok=False, reason="missing_target")
    if operation == "compare_first":
        if len(candidates) < 2:
            candidates = mentioned[:2]
        if len(candidates) < 2:
            return OpenProgram(operation, candidate_labels=candidates, ok=False, reason="missing_candidates")
    if operation == "unsupported":
        fallback = rule_open_program(question, labels)
        if fallback.ok:
            return fallback
        return OpenProgram("unsupported", ok=False, reason="unsupported")
    return OpenProgram(
        operation=operation,
        target_label=target,
        anchor_label=anchor,
        anchor_ordinal=ordinal,
        candidate_labels=candidates[:2],
        ok=True,
        reason="ok",
    )


def labels_mentioned(question: str, labels: Sequence[str]) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for label in labels:
        for variant in label_variants(label):
            match = re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(variant)}(?![A-Za-z0-9_])",
                question,
                flags=re.IGNORECASE,
            )
            if match is not None:
                found.append((match.start(), label))
                seen.add(label)
                break
    fuzzy = fuzzy_labels_mentioned(question, labels, seen)
    found.extend(fuzzy)
    return tuple(label for _pos, label in sorted(found))


def fuzzy_labels_mentioned(
    question: str,
    labels: Sequence[str],
    already_seen: set[str],
    *,
    threshold: float = 0.88,
) -> list[tuple[int, str]]:
    tokens = list(re.finditer(r"[A-Za-z0-9]+", question.lower()))
    if not tokens:
        return []
    result: list[tuple[int, str]] = []
    for label in labels:
        if label in already_seen:
            continue
        best: tuple[float, int] | None = None
        for variant in label_variants(label):
            variant_tokens = re.findall(r"[A-Za-z0-9]+", variant.lower())
            if not variant_tokens:
                continue
            variant_text = " ".join(variant_tokens)
            # Avoid fuzzy matching very short labels like "slam"; exact match
            # above is safer for those.
            if len(variant_text) < 6:
                continue
            token_count = len(variant_tokens)
            for width in {max(1, token_count - 1), token_count, token_count + 1}:
                if width > len(tokens):
                    continue
                for start_idx in range(0, len(tokens) - width + 1):
                    end_idx = start_idx + width
                    window = " ".join(match.group(0) for match in tokens[start_idx:end_idx])
                    score = SequenceMatcher(None, window, variant_text).ratio()
                    if score >= threshold and (best is None or score > best[0]):
                        best = (score, tokens[start_idx].start())
        if best is not None:
            result.append((best[1], label))
    return result


class EmbeddingTextParser:
    """Small local intent retriever over question templates.

    The embedding model is used for the semantic intent only.  Labels/ordinals
    are extracted after that so the downstream executor can still ground
    evidence spans.
    """

    def __init__(self, model_path: str, device_text: str = "auto") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(
            "cuda"
            if device_text == "auto" and torch.cuda.is_available()
            else "cpu" if device_text == "auto" else device_text
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device).eval()

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        return torch.nn.functional.normalize(pooled.float().cpu(), dim=1)


INTENT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "list_events": (
        "what sounds are in this audio",
        "list all events in the clip",
        "which audio events can be heard",
        "describe the sounds present",
    ),
    "exists": (
        "is there a target sound",
        "do you hear the target event",
        "is the target noise present",
        "can you hear any target sound",
    ),
    "count": (
        "how many times does the target sound occur",
        "count the number of target events",
        "how many target sounds are there",
    ),
    "first_event": (
        "what is the first sound",
        "which event starts first",
        "what begins earliest in the audio",
    ),
    "last_event": (
        "what is the last sound",
        "which event starts last",
        "what begins latest in the audio",
    ),
    "longest_event": (
        "which sound lasts the longest",
        "what event has the longest duration",
        "which audio event continues for the most time",
    ),
    "after": (
        "what sound starts after the anchor sound",
        "what happens right after the anchor event",
        "which event begins next after the anchor",
    ),
    "before": (
        "what sound starts before the anchor sound",
        "what happens right before the anchor event",
        "which event begins immediately before the anchor",
    ),
    "between": (
        "what sound occurs between two anchor events",
        "what sound is between the third target sound and another target sound",
        "what is between the anchor event and the boundary event",
        "which sound happens in the interval between two events",
        "which event happens between the first boundary and the second boundary",
        "what audio event is between the left sound and the right sound",
    ),
    "compare_first": (
        "between two sounds which starts earlier",
        "which of these two events begins first",
        "compare which sound has the earlier onset",
    ),
}


def label_text(label: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_/]+", " ", label)).strip().lower()


def ordinal_word(value: int) -> str:
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
    return words.get(value, str(value))


def dynamic_intent_templates(
    labels: Sequence[str],
    mentioned: Sequence[str],
    ordinal: int,
) -> dict[str, tuple[str, ...]]:
    if not mentioned:
        return {}
    first = label_text(mentioned[0])
    ord_text = ordinal_word(ordinal)
    result: dict[str, tuple[str, ...]] = {
        "exists": (
            f"is there a {first} sound",
            f"can you hear {first}",
            f"does {first} occur in the audio",
        ),
        "count": (
            f"how many times does {first} occur",
            f"count {first} sounds",
        ),
        "after": (
            f"what starts right after the {ord_text} {first} sound",
            f"what sound happens after the {ord_text} {first}",
        ),
        "before": (
            f"what starts right before the {ord_text} {first} sound",
            f"what sound happens before the {ord_text} {first}",
        ),
    }
    if len(mentioned) >= 2:
        second = label_text(mentioned[1])
        result["between"] = (
            f"what sound between the {ord_text} {first} sound and {second}",
            f"what sound is between the {ord_text} {first} and {second}",
            f"which event happens between {first} and {second}",
        )
        result["compare_first"] = (
            f"between {first} and {second} which starts earlier",
            f"which begins first {first} or {second}",
        )
    return result


def embedding_program(
    question: str,
    labels: Sequence[str],
    parser: EmbeddingTextParser,
) -> tuple[OpenProgram, dict[str, Any]]:
    mentioned = labels_mentioned(question, labels)
    ordinal = ordinal_from_question(question) or 1
    template_texts: list[str] = []
    template_ops: list[str] = []
    template_bank: dict[str, tuple[str, ...]] = dict(INTENT_TEMPLATES)
    for operation, templates in dynamic_intent_templates(labels, mentioned, ordinal).items():
        template_bank[operation] = (*template_bank.get(operation, ()), *templates)
    for operation, templates in template_bank.items():
        for template in templates:
            template_texts.append(template)
            template_ops.append(operation)
    vectors = parser.encode([question, *template_texts])
    query = vectors[0:1]
    template_vectors = vectors[1:]
    scores = (query @ template_vectors.T).squeeze(0)
    best_index = int(torch.argmax(scores).item())
    operation = template_ops[best_index]
    best_score = float(scores[best_index].item())

    cue = relation_cue(question)
    if cue in {"after", "before"}:
        operation = cue
    if "between" in question.lower() and len(mentioned) >= 2:
        operation = "between"

    if operation in {"list_events", "first_event", "last_event", "longest_event"}:
        program = OpenProgram(operation, reason=f"embedding:{best_score:.3f}")
    elif operation in {"exists", "count"}:
        program = OpenProgram(
            operation,
            target_label=mentioned[0] if mentioned else None,
            ok=bool(mentioned),
            reason=f"embedding:{best_score:.3f}" if mentioned else "embedding_missing_target",
        )
    elif operation in {"after", "before"}:
        anchor = mentioned[0] if mentioned else None
        program = OpenProgram(
            operation,
            anchor_label=anchor,
            anchor_ordinal=ordinal,
            ok=anchor is not None,
            reason=f"embedding:{best_score:.3f}" if anchor is not None else "embedding_missing_anchor",
        )
    elif operation == "between":
        program = OpenProgram(
            operation,
            target_label=mentioned[1] if len(mentioned) >= 2 else None,
            anchor_label=mentioned[0] if mentioned else None,
            anchor_ordinal=ordinal,
            candidate_labels=mentioned[:2],
            ok=len(mentioned) >= 2,
            reason=f"embedding:{best_score:.3f}" if len(mentioned) >= 2 else "embedding_missing_between_boundaries",
        )
    elif operation == "compare_first":
        program = OpenProgram(
            operation,
            candidate_labels=mentioned[:2],
            ok=len(mentioned) >= 2,
            reason=f"embedding:{best_score:.3f}" if len(mentioned) >= 2 else "embedding_missing_candidates",
        )
    else:
        program = OpenProgram("unsupported", ok=False, reason=f"embedding:{best_score:.3f}")

    topk = min(5, len(template_texts))
    ranked = torch.topk(scores, k=topk)
    metadata = {
        "parser": "embedding",
        "best_score": round(best_score, 4),
        "mentioned_labels": list(mentioned),
        "top_intents": [
            {
                "operation": template_ops[int(idx.item())],
                "template": template_texts[int(idx.item())],
                "score": round(float(value.item()), 4),
            }
            for value, idx in zip(ranked.values, ranked.indices)
        ],
    }
    return program, metadata


def rule_open_program(question: str, labels: Sequence[str]) -> OpenProgram:
    lowered = question.lower()
    mentioned = labels_mentioned(question, labels)
    cue = relation_cue(question)
    if any(token in lowered for token in ("what sounds", "which sounds", "what events", "which events", "list")):
        return OpenProgram("list_events")
    if "how many" in lowered or "count" in lowered:
        return OpenProgram("count", target_label=mentioned[0] if mentioned else None, ok=bool(mentioned), reason="rule")
    if lowered.startswith("is there") or lowered.startswith("are there") or "do you hear" in lowered:
        return OpenProgram("exists", target_label=mentioned[0] if mentioned else None, ok=bool(mentioned), reason="rule")
    if "longest" in lowered:
        return OpenProgram("longest_event")
    if "between" in lowered and len(mentioned) >= 2:
        ordinal = ordinal_from_question(question) or 1
        return OpenProgram(
            "between",
            target_label=mentioned[1],
            anchor_label=mentioned[0],
            anchor_ordinal=ordinal,
            candidate_labels=mentioned[:2],
            reason="rule",
        )
    if ("first" in lowered or "earliest" in lowered) and len(mentioned) < 2 and cue is None:
        return OpenProgram("first_event")
    if ("last" in lowered or "latest" in lowered) and cue is None:
        return OpenProgram("last_event")
    if cue in {"after", "before"}:
        anchor, ordinal = infer_anchor_from_question(question, labels)
        return OpenProgram(cue, anchor_label=anchor, anchor_ordinal=ordinal or 1, ok=anchor is not None, reason="rule")
    if len(mentioned) >= 2 and any(token in lowered for token in ("earlier", "sooner", "first", "starts before", "begins before")):
        return OpenProgram("compare_first", candidate_labels=mentioned[:2], reason="rule")
    return OpenProgram("unsupported", ok=False, reason="rule_unsupported")


def parse_open_question(
    question: str,
    labels: Sequence[str],
    llm: QwenTextParser | None,
    *,
    embedding_parser: EmbeddingTextParser | None = None,
) -> tuple[OpenProgram, str, Any]:
    if embedding_parser is not None:
        program, metadata = embedding_program(question, labels, embedding_parser)
        return program, json.dumps(metadata, ensure_ascii=False), metadata
    if llm is None:
        program = rule_open_program(question, labels)
        return program, "", None
    raw = llm.parse(open_parser_prompt(question, labels))
    obj = extract_json(raw)
    if obj is None:
        # Qwen sometimes emits a Python dict with single quotes.  The shared
        # extractor already tries literal_eval inside braces; this is a final
        # whole-string fallback.
        try:
            maybe = ast.literal_eval(raw.strip())
            obj = maybe if isinstance(maybe, dict) else None
        except (SyntaxError, ValueError):
            obj = None
    program = normalize_program(obj, labels, question)
    return program, raw, obj


def occurrences(events: Sequence[EventProposal], label: str) -> list[EventProposal]:
    return sorted(
        (event for event in events if event.label == label),
        key=lambda event: (event.onset_seconds, event.offset_seconds, -event.confidence),
    )


def unique_labels_chronological(events: Sequence[EventProposal]) -> list[str]:
    result: list[str] = []
    for event in sorted(events, key=lambda item: (item.onset_seconds, item.offset_seconds)):
        if event.label not in result:
            result.append(event.label)
    return result


def execute(program: OpenProgram, events: Sequence[EventProposal], max_list_events: int) -> OpenAnswer:
    ordered = sorted(events, key=lambda item: (item.onset_seconds, item.offset_seconds, -item.confidence))
    if not program.ok:
        return OpenAnswer("unsupported", True, (), program.reason)
    if not ordered:
        return OpenAnswer("no_evidence", True, (), "empty_inventory")

    if program.operation == "list_events":
        labels = unique_labels_chronological(ordered)[:max_list_events]
        return OpenAnswer(", ".join(labels) if labels else "no_evidence", not bool(labels), tuple(ordered[:max_list_events]), "chronological_unique_labels")

    if program.operation == "exists":
        occ = occurrences(ordered, str(program.target_label))
        return OpenAnswer("yes" if occ else "no", not bool(occ), tuple(occ[:max_list_events]), "target_presence")

    if program.operation == "count":
        occ = occurrences(ordered, str(program.target_label))
        return OpenAnswer(str(len(occ)), not bool(occ), tuple(occ[:max_list_events]), "target_occurrence_count")

    if program.operation == "first_event":
        event = ordered[0]
        return OpenAnswer(event.label, False, (event,), "earliest_predicted_onset")

    if program.operation == "last_event":
        event = max(ordered, key=lambda item: item.onset_seconds)
        return OpenAnswer(event.label, False, (event,), "latest_predicted_onset")

    if program.operation == "longest_event":
        event = max(ordered, key=lambda item: (item.duration_seconds, item.confidence))
        return OpenAnswer(event.label, False, (event,), "longest_predicted_duration")

    if program.operation in {"after", "before"}:
        anchor_events = occurrences(ordered, str(program.anchor_label))
        if len(anchor_events) < program.anchor_ordinal:
            return OpenAnswer("no_evidence", True, (), "anchor_occurrence_missing")
        anchor = anchor_events[program.anchor_ordinal - 1]
        if program.operation == "after":
            candidates = [
                event
                for event in ordered
                if event.onset_seconds > anchor.onset_seconds + 1e-6
                and event is not anchor
                and event.label != anchor.label
            ]
            if not candidates:
                return OpenAnswer("no_evidence", True, (anchor,), "no_following_event")
            nonoverlapping = [
                event
                for event in candidates
                if event.onset_seconds >= anchor.offset_seconds - 0.08
            ]
            confident = [event for event in nonoverlapping if event.confidence >= 0.50]
            if confident:
                answer = min(confident, key=lambda item: (item.onset_seconds, -item.confidence))
            elif nonoverlapping:
                answer = max(nonoverlapping, key=lambda item: item.confidence)
            else:
                answer = max(candidates, key=lambda item: item.confidence)
        else:
            candidates = [event for event in ordered if event.onset_seconds < anchor.onset_seconds - 1e-6 and event is not anchor]
            if not candidates:
                return OpenAnswer("no_evidence", True, (anchor,), "no_preceding_event")
            nonoverlapping = [
                event
                for event in candidates
                if event.offset_seconds <= anchor.onset_seconds + 0.08
            ]
            if nonoverlapping:
                answer = max(
                    nonoverlapping,
                    key=lambda item: (item.offset_seconds, item.confidence),
                )
            else:
                answer = max(candidates, key=lambda item: (item.onset_seconds, item.confidence))
        evidence = tuple(sorted((anchor, answer), key=lambda item: item.onset_seconds))
        return OpenAnswer(answer.label, False, evidence, f"{program.operation}_neighbor")

    if program.operation == "between":
        left_events = occurrences(ordered, str(program.anchor_label))
        if len(left_events) < program.anchor_ordinal:
            return OpenAnswer("no_evidence", True, (), "left_boundary_occurrence_missing")
        left = left_events[program.anchor_ordinal - 1]
        right_events = [
            event
            for event in occurrences(ordered, str(program.target_label))
            if event.onset_seconds > left.onset_seconds + 1e-6
        ]
        if not right_events:
            return OpenAnswer("no_evidence", True, (left,), "right_boundary_missing_after_left")
        right = min(right_events, key=lambda item: item.onset_seconds)
        lo = min(left.onset_seconds, right.onset_seconds)
        hi = max(left.onset_seconds, right.onset_seconds)
        boundary_labels = {str(program.anchor_label), str(program.target_label)}
        between_events = [
            event
            for event in ordered
            if event not in {left, right}
            and event.label not in boundary_labels
            and event.onset_seconds > lo + 1e-6
            and event.onset_seconds < hi - 1e-6
        ]
        evidence = tuple(sorted((left, *between_events, right), key=lambda item: item.onset_seconds))
        if not between_events:
            return OpenAnswer("no_evidence", True, evidence, "no_event_between_boundaries")
        # Most demo questions are singular ("what sound is between ...").  Use
        # the strongest non-boundary proposal as the answer to avoid listing
        # low-confidence aliases/noise such as "Rattle" next to "Keys_jangling".
        answer = max(between_events, key=lambda item: item.confidence)
        evidence = tuple(sorted((left, answer, right), key=lambda item: item.onset_seconds))
        return OpenAnswer(answer.label, False, evidence, "strongest_event_between_boundaries")

    if program.operation == "compare_first":
        firsts: list[EventProposal] = []
        for label in program.candidate_labels[:2]:
            occ = occurrences(ordered, label)
            if not occ:
                return OpenAnswer("no_evidence", True, tuple(firsts), f"candidate_missing:{label}")
            firsts.append(occ[0])
        evidence = tuple(sorted(firsts, key=lambda item: item.onset_seconds))
        return OpenAnswer(evidence[0].label, False, evidence, "earliest_among_candidates")

    return OpenAnswer("unsupported", True, (), f"unsupported_operation:{program.operation}")


def event_payload(event: EventProposal) -> dict[str, Any]:
    return {
        "label": event.label,
        "onset_seconds": round(float(event.onset_seconds), 3),
        "offset_seconds": round(float(event.offset_seconds), 3),
        "confidence": round(float(event.confidence), 4),
    }


def answer_label_set(answer_text: str) -> set[str]:
    lowered = answer_text.strip().lower()
    if lowered in {"", "yes", "no", "no_evidence", "unsupported"}:
        return set()
    return {piece.strip() for piece in answer_text.split(",") if piece.strip()}


def split_answer_context_events(answer: OpenAnswer) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = answer_label_set(answer.answer)
    answer_events: list[dict[str, Any]] = []
    context_events: list[dict[str, Any]] = []
    for event in answer.evidence:
        payload = event_payload(event)
        if event.label in labels:
            answer_events.append(payload)
        else:
            context_events.append(payload)
    return answer_events, context_events


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
    parser_mode = "rule" if args.no_llm else args.parser
    llm = QwenTextParser(args.model, max_new_tokens=120) if parser_mode == "qwen" else None
    embedding_parser = (
        EmbeddingTextParser(args.embedding_model, device_text=args.device)
        if parser_mode == "embedding"
        else None
    )
    program, raw, raw_json = parse_open_question(
        args.question,
        labels,
        llm,
        embedding_parser=embedding_parser,
    )
    answer = execute(program, inventory, args.max_list_events)
    answer_events, context_events = split_answer_context_events(answer)
    payload = {
        "format": "qces_open_event_qa_probe_v1",
        "scene_id": scene_id,
        "question": args.question,
        "threshold": args.threshold,
        "available_label_count": len(labels),
        "inventory_event_count": len(inventory),
        "program": asdict(program),
        "parser": parser_mode,
        "llm_raw": raw,
        "llm_json": raw_json,
        "answer": answer.answer,
        "no_evidence": answer.no_evidence,
        "answer_reason": answer.reason,
        "answer_events": answer_events,
        "context_events": context_events,
        "evidence": [event_payload(event) for event in answer.evidence],
        "inventory_preview": [event_payload(event) for event in inventory[: args.max_list_events]],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
