"""Qwen-backed structured parser for QCES question text.

This module is intentionally separate from ``question_parsing.py``.  The
existing rule parser remains the registered template parser; this module is an
optional LLM parser used to test whether a local Qwen-family model can recover
the same structured query from less templated question wording.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mixi_understanding.qces.question_parsing import (
    NO_EVIDENCE_OPTION,
    ParsedQuestion,
    RELATION_AFTER,
    RELATION_BEFORE,
    RELATION_FIRST,
    parse_question,
)


RELATIONS = (RELATION_AFTER, RELATION_BEFORE, RELATION_FIRST)
RELATION_ALIASES: Mapping[str, str] = {
    "next": RELATION_AFTER,
    "following": RELATION_AFTER,
    "subsequent": RELATION_AFTER,
    "later": RELATION_AFTER,
    "previous": RELATION_BEFORE,
    "preceding": RELATION_BEFORE,
    "prior": RELATION_BEFORE,
    "earliest": RELATION_FIRST,
    "sooner": RELATION_FIRST,
    "first_event": RELATION_FIRST,
}
ORDINAL_WORDS: Mapping[str, int] = {
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


@dataclass(frozen=True)
class QwenParseResult:
    """One Qwen parser response after JSON extraction and schema validation."""

    parsed: ParsedQuestion
    valid_json: bool
    schema_valid: bool
    raw_text: str
    normalized_payload: dict[str, Any]
    reason: str


def _label_key(value: str) -> str:
    lowered = value.strip().lower()
    lowered = lowered.replace("-", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9]+", "", lowered)


def canonical_label(value: Any, taxonomy: Sequence[str]) -> str | None:
    """Map model-emitted labels back to the exact benchmark label string."""

    if not isinstance(value, str) or not value.strip():
        return None
    exact = value.strip()
    if exact in taxonomy:
        return exact
    without_parenthetical = re.sub(r"\s*\([^)]*\)\s*$", "", exact).strip()
    if without_parenthetical in taxonomy:
        return without_parenthetical
    table = {_label_key(label): label for label in taxonomy}
    return table.get(_label_key(exact)) or table.get(_label_key(without_parenthetical))


def canonical_ordinal(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        result = int(value)
        return result if result > 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text.isdigit():
        result = int(text)
        return result if result > 0 else None
    if text in ORDINAL_WORDS:
        return ORDINAL_WORDS[text]
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", text)
    if match is not None:
        result = int(match.group(1))
        return result if result > 0 else None
    return None


def ordinal_in_text(value: Any) -> int | None:
    """Find an ordinal mention inside a longer phrase."""

    if not isinstance(value, str):
        return None
    lowered = value.lower()
    for word, ordinal in ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return ordinal
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", lowered)
    if match is not None:
        result = int(match.group(1))
        return result if result > 0 else None
    return None


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Extract the first JSON object from a free-form generation."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload, "whole_text_json"
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        return None, "no_open_brace"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    return None, f"json_decode_error:{exc.msg}"
                if isinstance(payload, dict):
                    return payload, "substring_json"
                return None, "json_not_object"
    return None, "unclosed_json_object"


def extract_partial_json_fields(text: str) -> tuple[dict[str, Any] | None, str]:
    """Recover enough fields from a truncated JSON object for after/before."""

    relation_match = re.search(r'"relation"\s*:\s*"([^"]+)"', text)
    anchor_match = re.search(r'"anchor_label"\s*:\s*(null|"[^"]*")', text)
    ordinal_match = re.search(
        r'"anchor_ordinal"\s*:\s*(null|-?\d+|"[^"]*")', text
    )
    if relation_match is None:
        return None, "partial_missing_relation"
    payload: dict[str, Any] = {
        "relation": relation_match.group(1),
        "candidate_labels": [],
    }
    if anchor_match is not None:
        raw = anchor_match.group(1)
        payload["anchor_label"] = None if raw == "null" else raw.strip('"')
    if ordinal_match is not None:
        raw = ordinal_match.group(1)
        if raw == "null":
            payload["anchor_ordinal"] = None
        elif raw.startswith('"'):
            payload["anchor_ordinal"] = raw.strip('"')
        else:
            payload["anchor_ordinal"] = int(raw)
    return payload, "partial_json_fields"


def normalize_qwen_payload(
    payload: Mapping[str, Any],
    *,
    question: str,
    answer_options: Sequence[str],
    taxonomy: Sequence[str],
) -> tuple[ParsedQuestion, dict[str, Any], bool, str]:
    """Validate a Qwen JSON object and convert it to ``ParsedQuestion``."""

    option_labels = tuple(
        option for option in answer_options if option != NO_EVIDENCE_OPTION
    )
    relation_raw = payload.get("relation")
    relation = relation_raw.strip().lower() if isinstance(relation_raw, str) else None
    relation = RELATION_ALIASES.get(relation or "", relation)
    if relation not in RELATIONS:
        parsed = ParsedQuestion(
            relation=None,
            anchor_label=None,
            anchor_ordinal=None,
            candidate_labels=(),
            option_labels=option_labels,
            mentioned_labels=(),
            ok=False,
            reason="qwen_invalid_relation",
        )
        return parsed, {"relation": relation_raw}, False, "invalid_relation"

    anchor_raw = payload.get("anchor_label")
    anchor = canonical_label(anchor_raw, taxonomy)
    ordinal = (
        ordinal_in_text(anchor_raw)
        or canonical_ordinal(payload.get("anchor_ordinal"))
        or ordinal_in_text(question)
    )
    candidates_raw = payload.get("candidate_labels", [])
    if not isinstance(candidates_raw, list):
        candidates_raw = []
    candidates = tuple(
        dict.fromkeys(
            label
            for label in (
                canonical_label(item, taxonomy) for item in candidates_raw
            )
            if label is not None
        )
    )
    if relation in {RELATION_AFTER, RELATION_BEFORE} and anchor is None:
        option_set = set(option_labels)
        non_option_candidates = [
            label for label in candidates if label not in option_set
        ]
        if len(non_option_candidates) == 1:
            anchor = non_option_candidates[0]

    mentioned: list[str] = []
    if anchor is not None:
        mentioned.append(anchor)
    for label in candidates:
        if label not in mentioned:
            mentioned.append(label)

    schema_valid = True
    reason = "qwen_schema_valid"
    if relation == RELATION_FIRST:
        if len(candidates) != 2:
            schema_valid = False
            reason = "first_requires_two_candidate_labels"
        parsed = ParsedQuestion(
            relation=relation,
            anchor_label=None,
            anchor_ordinal=1,
            candidate_labels=candidates[:2],
            option_labels=option_labels,
            mentioned_labels=tuple(mentioned),
            ok=schema_valid,
            reason=reason,
        )
    else:
        if anchor is None:
            schema_valid = False
            reason = "after_before_requires_anchor_label"
        elif ordinal is None:
            schema_valid = False
            reason = "after_before_requires_anchor_ordinal"
        parsed = ParsedQuestion(
            relation=relation,
            anchor_label=anchor,
            anchor_ordinal=ordinal,
            candidate_labels=(),
            option_labels=option_labels,
            mentioned_labels=tuple(mentioned),
            ok=schema_valid,
            reason=reason,
        )

    normalized = {
        "relation": relation,
        "anchor_label": anchor,
        "anchor_ordinal": ordinal,
        "candidate_labels": list(candidates),
    }
    return parsed, normalized, schema_valid, reason


def result_from_generation(
    text: str,
    *,
    question: str,
    answer_options: Sequence[str],
    taxonomy: Sequence[str],
    fallback_to_rule: bool,
) -> QwenParseResult:
    """Convert raw model text to a parse result, optionally with rule fallback."""

    payload, json_reason = extract_json_object(text)
    partial_reason: str | None = None
    if payload is None:
        payload, partial_reason = extract_partial_json_fields(text)

    if payload is not None:
        parsed, normalized, schema_valid, reason = normalize_qwen_payload(
            payload,
            question=question,
            answer_options=answer_options,
            taxonomy=taxonomy,
        )
        if schema_valid:
            return QwenParseResult(
                parsed=parsed,
                valid_json=partial_reason is None,
                schema_valid=True,
                raw_text=text,
                normalized_payload=normalized,
                reason=reason if partial_reason is None else partial_reason,
            )
        if not fallback_to_rule:
            return QwenParseResult(
                parsed=parsed,
                valid_json=partial_reason is None,
                schema_valid=False,
                raw_text=text,
                normalized_payload=normalized,
                reason=reason,
            )
    else:
        normalized = {}
        reason = json_reason

    if fallback_to_rule:
        parsed = parse_question(question, answer_options, taxonomy)
        return QwenParseResult(
            parsed=parsed,
            valid_json=payload is not None,
            schema_valid=False,
            raw_text=text,
            normalized_payload=dict(normalized),
            reason=f"fallback_rule_after_{reason}",
        )

    parsed = ParsedQuestion(
        relation=None,
        anchor_label=None,
        anchor_ordinal=None,
        candidate_labels=(),
        option_labels=tuple(
            option for option in answer_options if option != NO_EVIDENCE_OPTION
        ),
        mentioned_labels=(),
        ok=False,
        reason=f"qwen_{reason}",
    )
    return QwenParseResult(
        parsed=parsed,
        valid_json=False,
        schema_valid=False,
        raw_text=text,
        normalized_payload={},
        reason=reason,
    )


def qwen_parser_prompt(
    *,
    question: str,
    answer_options: Sequence[str],
    taxonomy: Sequence[str],
) -> str:
    """Render the deterministic instruction prompt for JSON parsing."""

    labels = ", ".join(taxonomy)
    options = ", ".join(answer_options)
    return (
        "You are a strict parser for an audio question answering benchmark.\n"
        "Return only one JSON object. Do not explain.\n"
        "Schema:\n"
        "{"
        '"relation":"after|before|first",'
        '"anchor_label":"exact label or null",'
        '"anchor_ordinal":"positive integer or null",'
        '"candidate_labels":["exact label","exact label"]'
        "}\n"
        "Rules:\n"
        "- For after/before questions, identify the anchor event label and which occurrence of it is referenced. Set candidate_labels to [].\n"
        "- For first/earliest questions, identify the two compared event labels. Set anchor_label and anchor_ordinal to null.\n"
        "- In phrases like 'after the first Camera' or 'before the second Camera', Camera is the anchor_label and first/second is anchor_ordinal.\n"
        "- 'next onset', 'subsequent onset', 'following', and 'later' mean relation='after'.\n"
        "- 'previous onset', 'preceding onset', and 'earlier' mean relation='before'.\n"
        "- Do not put answer options in candidate_labels unless the question asks which of two named labels starts first.\n"
        "- Use exact labels from Allowed labels or Answer options. Keep underscores and punctuation exactly.\n"
        "- The word 'event' means sound event onset.\n"
        "Examples:\n"
        'Question: Which event has the subsequent onset after the first Camera?\n'
        'JSON: {"relation":"after","anchor_label":"Camera","anchor_ordinal":1,"candidate_labels":[]}\n'
        'Question: What begins just before the second Camera begins?\n'
        'JSON: {"relation":"before","anchor_label":"Camera","anchor_ordinal":2,"candidate_labels":[]}\n'
        'Question: Find occurrence #2 of Camera; which sound has the immediately previous onset?\n'
        'JSON: {"relation":"before","anchor_label":"Camera","anchor_ordinal":2,"candidate_labels":[]}\n'
        'Question: Find occurrence #1 of Microwave_oven; which sound has the next onset?\n'
        'JSON: {"relation":"after","anchor_label":"Microwave_oven","anchor_ordinal":1,"candidate_labels":[]}\n'
        'Question: Between Bark and Meow, which sound starts earlier?\n'
        'JSON: {"relation":"first","anchor_label":null,"anchor_ordinal":null,"candidate_labels":["Bark","Meow"]}\n'
        f"Allowed labels: {labels}\n"
        f"Answer options: {options}\n"
        f"Question: {question}\n"
        "JSON:"
    )
