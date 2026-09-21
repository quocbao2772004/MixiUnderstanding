"""Symbolic relational planning over event proposals.

The planner is the part of QCES-v6 that is *not* learned.  Given an inventory
of ``(label, onset, offset, confidence)`` proposals and a parsed question it
resolves the ordinal and the temporal relation with the same rules the
benchmark used to define its answers, and returns both the answer label and the
minimal set of events a listener needs in order to check that answer.

Making this step symbolic is the central design claim of the method: an
``after``/``before``/``first`` question is a query over event identity and
occurrence order, and a dense frame-level gate has no representation for either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mixi_understanding.qces.event_proposals import EventProposal
from mixi_understanding.qces.question_parsing import (
    RELATION_AFTER,
    RELATION_BEFORE,
    RELATION_FIRST,
    ParsedQuestion,
)

ONSET_TIE_SECONDS = 1e-6


@dataclass(frozen=True)
class EvidencePlan:
    """What to separate, and the answer that separation is evidence for."""

    no_evidence: bool
    answer_label: str | None
    evidence: tuple[EventProposal, ...]
    reason: str

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.label for item in self.evidence))

    @property
    def spans(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (item.onset_seconds, item.offset_seconds) for item in self.evidence
        )


def _no_evidence(reason: str) -> EvidencePlan:
    return EvidencePlan(
        no_evidence=True, answer_label=None, evidence=(), reason=reason
    )


def _occurrences(
    proposals: Sequence[EventProposal], label: str
) -> list[EventProposal]:
    return sorted(
        (item for item in proposals if item.label == label),
        key=lambda item: (item.onset_seconds, item.offset_seconds),
    )


def plan_from_proposals(
    parsed: ParsedQuestion, proposals: Sequence[EventProposal]
) -> EvidencePlan:
    """Resolve the question against a predicted event inventory."""

    if not parsed.ok or parsed.relation is None:
        return _no_evidence(f"unparsed_question:{parsed.reason}")

    if parsed.relation == RELATION_FIRST:
        if len(parsed.candidate_labels) < 2:
            return _no_evidence("first_relation_needs_two_candidates")
        earliest: list[EventProposal] = []
        for label in parsed.candidate_labels:
            occurrences = _occurrences(proposals, label)
            if not occurrences:
                return _no_evidence(f"absent_first_candidate:{label}")
            earliest.append(occurrences[0])
        ordered = sorted(earliest, key=lambda item: item.onset_seconds)
        return EvidencePlan(
            no_evidence=False,
            answer_label=ordered[0].label,
            evidence=tuple(ordered),
            reason="earliest_onset_among_named_candidates",
        )

    if parsed.anchor_label is None or parsed.anchor_ordinal is None:
        return _no_evidence("missing_anchor_specification")

    occurrences = _occurrences(proposals, parsed.anchor_label)
    if len(occurrences) < parsed.anchor_ordinal:
        return _no_evidence(
            f"absent_anchor_occurrence:{parsed.anchor_label}"
            f"#{parsed.anchor_ordinal}/{len(occurrences)}"
        )
    anchor = occurrences[parsed.anchor_ordinal - 1]

    # Neighbour search runs over every proposed occurrence, including further
    # occurrences of the anchor label itself: the benchmark orders adjacent
    # events by onset over unique semantic onsets, not over distinct classes.
    others = [item for item in proposals if item is not anchor]
    if parsed.relation == RELATION_AFTER:
        following = [
            item
            for item in others
            if item.onset_seconds > anchor.onset_seconds + ONSET_TIE_SECONDS
        ]
        if not following:
            return _no_evidence("no_following_proposal")
        answer = min(following, key=lambda item: item.onset_seconds)
    elif parsed.relation == RELATION_BEFORE:
        preceding = [
            item
            for item in others
            if item.onset_seconds < anchor.onset_seconds - ONSET_TIE_SECONDS
        ]
        if not preceding:
            return _no_evidence("no_preceding_proposal")
        answer = max(preceding, key=lambda item: item.onset_seconds)
    else:
        return _no_evidence(f"unsupported_relation:{parsed.relation}")

    evidence = tuple(
        sorted((anchor, answer), key=lambda item: item.onset_seconds)
    )
    return EvidencePlan(
        no_evidence=False,
        answer_label=answer.label,
        evidence=evidence,
        reason="anchor_occurrence_and_temporal_neighbour",
    )


def prompt_for_labels(labels: Sequence[str], describe) -> str:
    """Render the canonical separator prompt for a planned label set."""

    phrases = [describe(label).replace("_", " ") for label in labels]
    if not phrases:
        return ""
    if len(phrases) == 1:
        joined = phrases[0]
    else:
        joined = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return f"the sounds of {joined}"
