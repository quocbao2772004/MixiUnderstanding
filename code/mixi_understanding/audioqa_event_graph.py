"""Auditable multi-intent AudioQA over a predicted temporal event inventory."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AudioQAResult:
    supported: bool
    intent: str
    answer: str
    mentioned_labels: tuple[str, ...]
    evidence: tuple[Mapping[str, Any], ...]
    reason: str

    @property
    def evidence_window(self) -> tuple[float, float] | None:
        if not self.evidence:
            return None
        return (
            min(float(event["start_seconds"]) for event in self.evidence),
            max(float(event["end_seconds"]) for event in self.evidence),
        )


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.replace("_", " "))
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    without_marks = without_marks.lower().replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def _mentions(question: str, labels: Sequence[str]) -> tuple[str, ...]:
    normalized = normalize_text(question)
    alias_targets: dict[str, set[str]] = {}
    for label in labels:
        aliases = {normalize_text(label)}
        without_parenthetical = re.sub(r"\([^)]*\)", " ", label)
        aliases.add(normalize_text(without_parenthetical))
        for alias in aliases:
            if alias:
                alias_targets.setdefault(alias, set()).add(label)
    found: list[tuple[int, int, str]] = []
    for alias, targets in alias_targets.items():
        # Shortened aliases are accepted only when unique in the declared
        # ontology. This supports "frying" -> "Frying (food)" without making
        # ambiguous words such as "speech" silently choose one class.
        if len(targets) != 1:
            continue
        match = re.search(rf"(?:^| ){re.escape(alias)}(?: |$)", normalized)
        if match is not None:
            found.append((match.start(), -len(alias), next(iter(targets))))
    found.sort()
    result: list[str] = []
    for _, _, label in found:
        if label not in result:
            result.append(label)
    return tuple(result)


def _contains(text: str, cues: Sequence[str]) -> bool:
    padded = f" {normalize_text(text)} "
    return any(f" {normalize_text(cue)} " in padded for cue in cues)


def _flatten(inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in inventory:
        for occurrence_index, occurrence in enumerate(item.get("occurrences", []), 1):
            events.append(
                dict(occurrence)
                | {
                    "label": str(item["label"]),
                    "display_label": str(item["display_label"]),
                    "class_score": float(item["score"]),
                    "occurrence_index": occurrence_index,
                }
            )
    return sorted(
        events,
        key=lambda event: (
            float(event["start_seconds"]),
            float(event["end_seconds"]),
            str(event["label"]),
        ),
    )


def _display(label: str) -> str:
    return label.replace("_and_", " & ").replace("_", " ").strip()


def _events_for(events: Sequence[Mapping[str, Any]], label: str) -> list[Mapping[str, Any]]:
    return [event for event in events if str(event["label"]) == label]


def _unsupported(intent: str, mentions: tuple[str, ...], reason: str) -> AudioQAResult:
    return AudioQAResult(False, intent, "UNSUPPORTED", mentions, (), reason)


def answer_event_graph_question(
    question: str,
    inventory: Sequence[Mapping[str, Any]],
    taxonomy: Sequence[str],
) -> AudioQAResult:
    """Answer a supported question using predictions only, never gold events."""

    normalized = normalize_text(question)
    mentions = _mentions(question, taxonomy)
    events = _flatten(inventory)
    if not normalized:
        return _unsupported("unknown", mentions, "empty_question")

    before = _contains(question, ("before", "preceding", "prior", "trước"))
    after = _contains(question, ("after", "following", "sau"))
    between = _contains(question, ("between", "in between", "ở giữa", "giữa"))
    overlap = _contains(
        question,
        (
            "overlap",
            "overlaps",
            "overlapping",
            "at the same time",
            "đồng thời",
            "chồng",
        ),
    )
    count = _contains(question, ("how many", "number of", "bao nhiêu"))
    locate = _contains(
        question,
        ("where", "when", "what time", "timestamp", "đoạn nào", "ở đâu", "khi nào"),
    )
    exists = _contains(
        question,
        ("is there", "do you hear", "can you hear", "có tiếng", "có âm thanh"),
    )
    longest = _contains(question, ("longest", "dài nhất", "lâu nhất"))
    first = _contains(
        question,
        (
            "first",
            "earliest",
            "at the beginning",
            "at the start",
            "beginning of the audio",
            "start of the audio",
            "start of audio",
            "begin in this audio",
            "begins in this audio",
            "begin the audio",
            "begins the audio",
            "starts the audio",
            "opening sound",
            "đầu tiên",
            "sớm nhất",
            "ở đầu",
            "đầu đoạn audio",
        ),
    )
    last = _contains(
        question,
        (
            "last",
            "latest",
            "at the end",
            "end of the audio",
            "end of audio",
            "toward the end",
            "end in this audio",
            "ends in this audio",
            "end the audio",
            "ends the audio",
            "finishes the audio",
            "closing sound",
            "cuối cùng",
            "muộn nhất",
            "ở cuối",
            "cuối đoạn audio",
        ),
    )
    list_events = _contains(
        question,
        (
            "list the sounds",
            "list all sounds",
            "list the events",
            "what sounds are present",
            "which sounds are present",
            "what sounds can be heard",
            "which sounds can be heard",
            "what sounds are in this audio",
            "which sounds occur in this audio",
            "what events are in this audio",
            "những âm thanh",
            "các âm thanh",
            "có âm thanh gì",
        ),
    )

    if between:
        if len(mentions) < 2:
            return _unsupported("between", mentions, "between_needs_two_labels")
        anchors = []
        for label in mentions[:2]:
            occurrences = _events_for(events, label)
            if not occurrences:
                return AudioQAResult(True, "between", "NONE", mentions, (), "anchor_absent")
            anchors.append(occurrences[0])
        left, right = sorted(anchors, key=lambda event: float(event["start_seconds"]))
        answers = [
            event
            for event in events
            if float(left["start_seconds"])
            < float(event["start_seconds"])
            < float(right["start_seconds"])
            and event not in anchors
        ]
        labels = list(dict.fromkeys(str(event["display_label"]) for event in answers))
        return AudioQAResult(
            True,
            "between",
            ", ".join(labels) if labels else "NONE",
            mentions,
            tuple([left, *answers, right]),
            "events_between_anchor_onsets",
        )

    if overlap:
        anchors = _events_for(events, mentions[0]) if mentions else []
        if mentions and not anchors:
            return AudioQAResult(True, "overlap", "NONE", mentions, (), "anchor_absent")
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        candidates = anchors if anchors else events
        for anchor in candidates:
            for event in events:
                if event is anchor or str(event["label"]) == str(anchor["label"]):
                    continue
                if min(float(anchor["end_seconds"]), float(event["end_seconds"])) > max(
                    float(anchor["start_seconds"]), float(event["start_seconds"])
                ):
                    pairs.append((anchor, event))
        if not pairs:
            return AudioQAResult(True, "overlap", "NONE", mentions, tuple(anchors), "no_overlap")
        evidence: list[Mapping[str, Any]] = []
        answer_labels: list[str] = []
        for anchor, event in pairs:
            for item in (anchor, event):
                if item not in evidence:
                    evidence.append(item)
            label = str(event["display_label"])
            if label not in answer_labels:
                answer_labels.append(label)
        return AudioQAResult(
            True, "overlap", ", ".join(answer_labels), mentions, tuple(evidence), "interval_overlap"
        )

    if before or after:
        if before == after or not mentions:
            return _unsupported("before_after", mentions, "relation_needs_one_anchor")
        anchor_events = _events_for(events, mentions[0])
        if not anchor_events:
            return AudioQAResult(True, "before" if before else "after", "NONE", mentions, (), "anchor_absent")
        anchor = anchor_events[0]
        position = events.index(anchor)
        answer_index = position - 1 if before else position + 1
        if answer_index < 0 or answer_index >= len(events):
            return AudioQAResult(
                True,
                "before" if before else "after",
                "NONE",
                mentions,
                (anchor,),
                "no_temporal_neighbour",
            )
        answer = events[answer_index]
        return AudioQAResult(
            True,
            "before" if before else "after",
            str(answer["display_label"]),
            mentions,
            tuple(sorted((anchor, answer), key=lambda event: float(event["start_seconds"]))),
            "adjacent_predicted_onset",
        )

    if count:
        if not mentions:
            return _unsupported("count", mentions, "count_needs_label")
        selected = _events_for(events, mentions[0])
        return AudioQAResult(True, "count", str(len(selected)), mentions, tuple(selected), "occurrence_count")

    if locate:
        if not mentions:
            return _unsupported("locate", mentions, "locate_needs_label")
        selected = _events_for(events, mentions[0])
        if not selected:
            return AudioQAResult(True, "locate", "NONE", mentions, (), "event_absent")
        answer = ", ".join(
            f"{float(event['start_seconds']):.2f}–{float(event['end_seconds']):.2f}s"
            for event in selected
        )
        return AudioQAResult(True, "locate", answer, mentions, tuple(selected), "predicted_timestamps")

    if exists:
        if not mentions:
            return _unsupported("exists", mentions, "exists_needs_label")
        selected = _events_for(events, mentions[0])
        return AudioQAResult(
            True,
            "exists",
            "YES" if selected else "NO",
            mentions,
            tuple(selected),
            "predicted_inventory_membership",
        )

    if longest:
        if not events:
            return AudioQAResult(True, "longest", "NONE", mentions, (), "empty_inventory")
        answer = max(
            events,
            key=lambda event: float(event["end_seconds"]) - float(event["start_seconds"]),
        )
        return AudioQAResult(
            True, "longest", str(answer["display_label"]), mentions, (answer,), "maximum_duration"
        )

    if first or last:
        if first == last or not events:
            return _unsupported("first_last", mentions, "ambiguous_or_empty_inventory")
        answer = events[0] if first else events[-1]
        return AudioQAResult(
            True,
            "first" if first else "last",
            str(answer["display_label"]),
            mentions,
            (answer,),
            "onset_order",
        )

    if list_events:
        unique = list(dict.fromkeys(str(event["display_label"]) for event in events))
        return AudioQAResult(
            True,
            "list",
            ", ".join(unique) if unique else "NONE",
            mentions,
            tuple(events),
            "predicted_inventory",
        )

    return _unsupported("unknown", mentions, "unsupported_intent")
