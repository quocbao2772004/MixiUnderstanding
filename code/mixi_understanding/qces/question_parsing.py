"""Parse a QCES natural-language question into a structured relational query.

The evaluation protocol for QCES-v6 forbids reading the annotation fields
``relation``, ``query_label``, ``query_instance_ordinal`` and
``query_candidate_labels`` at inference time.  This module recovers those
fields from the question surface string plus two pieces of public benchmark
metadata: the answer options shown to the system and the declared label
taxonomy of the benchmark.  Neither is a per-record annotation.

The parser is deliberately rule-based and auditable.  Its accuracy against the
withheld annotation is reported as a benchmark statistic, so a template family
that the rules cannot read shows up as a measured parse failure instead of a
silent oracle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

NO_EVIDENCE_OPTION = "no_evidence"

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

# Surface cues.  ``before`` is tested first because several ``before``
# templates also contain an ``after``-style word ("starts last" plus "onset").
BEFORE_CUES: tuple[str, ...] = (
    "before",
    "preceding",
    "precedes",
    "predecessor",
    "prior",
    "ahead of",
    "earlier than",
)
AFTER_CUES: tuple[str, ...] = (
    "after",
    "following",
    "follows",
    "next",
    "successor",
    "subsequent",
    "once the",
)

RELATION_AFTER = "after"
RELATION_BEFORE = "before"
RELATION_FIRST = "first"


@dataclass(frozen=True)
class ParsedQuestion:
    """A structured query recovered from the question surface string."""

    relation: str | None
    anchor_label: str | None
    anchor_ordinal: int | None
    candidate_labels: tuple[str, ...]
    option_labels: tuple[str, ...]
    mentioned_labels: tuple[str, ...]
    ok: bool
    reason: str

    @property
    def query_labels(self) -> tuple[str, ...]:
        """Every label the separator may be asked about for this question."""

        ordered = list(self.mentioned_labels)
        for label in self.option_labels:
            if label not in ordered:
                ordered.append(label)
        return tuple(ordered)


def _label_pattern(label: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])")


def find_label_mentions(
    question: str, taxonomy: Iterable[str]
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Return labels mentioned in the question, ordered by first character."""

    positions: dict[str, int] = {}
    for label in taxonomy:
        match = _label_pattern(label).search(question)
        if match is not None:
            positions[label] = match.start()
    # A longer label can contain a shorter one as a proper substring only when
    # the shorter one is a full underscore-delimited token, which the pattern
    # boundary already rejects.  Sorting by position keeps mention order.
    ordered = tuple(sorted(positions, key=lambda label: (positions[label], label)))
    return ordered, positions


def _find_ordinal(question: str, before_index: int | None = None) -> int | None:
    lowered = question.lower()
    best: tuple[int, int] | None = None
    for word, value in ORDINAL_WORDS.items():
        for match in re.finditer(rf"\b{word}\b", lowered):
            if before_index is not None and match.start() > before_index:
                continue
            if best is None or match.start() > best[0]:
                best = (match.start(), value)
    return None if best is None else best[1]


def parse_question(
    question: str,
    answer_options: Sequence[str],
    taxonomy: Iterable[str],
) -> ParsedQuestion:
    """Recover relation, anchor and candidate labels from the question text."""

    options = tuple(
        option for option in answer_options if option != NO_EVIDENCE_OPTION
    )
    vocabulary = set(taxonomy) | set(options)
    mentions, positions = find_label_mentions(question, vocabulary)

    if len(mentions) >= 2:
        # Every ``first`` template compares exactly two named classes; the
        # ``after``/``before`` templates name only the anchor.
        return ParsedQuestion(
            relation=RELATION_FIRST,
            anchor_label=None,
            anchor_ordinal=1,
            candidate_labels=mentions[:2],
            option_labels=options,
            mentioned_labels=mentions,
            ok=True,
            reason="two_named_classes_imply_first_relation",
        )

    lowered = question.lower()
    if any(cue in lowered for cue in BEFORE_CUES):
        relation = RELATION_BEFORE
    elif any(cue in lowered for cue in AFTER_CUES):
        relation = RELATION_AFTER
    else:
        return ParsedQuestion(
            relation=None,
            anchor_label=mentions[0] if mentions else None,
            anchor_ordinal=None,
            candidate_labels=(),
            option_labels=options,
            mentioned_labels=mentions,
            ok=False,
            reason="no_relation_cue_in_question",
        )

    if not mentions:
        return ParsedQuestion(
            relation=relation,
            anchor_label=None,
            anchor_ordinal=None,
            candidate_labels=(),
            option_labels=options,
            mentioned_labels=(),
            ok=False,
            reason="no_anchor_label_mention",
        )

    anchor = mentions[0]
    ordinal = _find_ordinal(question, positions[anchor]) or _find_ordinal(question)
    if ordinal is None:
        return ParsedQuestion(
            relation=relation,
            anchor_label=anchor,
            anchor_ordinal=None,
            candidate_labels=(),
            option_labels=options,
            mentioned_labels=mentions,
            ok=False,
            reason="no_ordinal_in_question",
        )
    return ParsedQuestion(
        relation=relation,
        anchor_label=anchor,
        anchor_ordinal=ordinal,
        candidate_labels=(),
        option_labels=options,
        mentioned_labels=mentions,
        ok=True,
        reason="anchor_and_ordinal_recovered",
    )
