"""Qwen question-program contract and deterministic event-graph executor.

Qwen is used only for language understanding.  Answers and acoustic evidence
are still produced from the supplied temporal inventory, so the LLM cannot
invent an event that is not present in the detector output.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from mixi_understanding.audioqa_event_graph import AudioQAResult, normalize_text


SUPPORTED_INTENTS = (
    "list",
    "exists",
    "locate",
    "count",
    "first",
    "last",
    "longest",
    "before",
    "after",
    "between",
    "overlap",
    "speaker_count",
    "speaker_identity",
    "speech_content",
    "speech_before_quote",
    "speech_after_quote",
    "open_audioqa",
    "unsupported",
)


# Bilingual lexical descriptions are part of the ontology contract, not scene
# annotations.  They let Qwen ground Vietnamese wording to the same stable
# internal labels used by the detector and executor.
KNOWN_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "Accelerating_and_revving": (
        "accelerating", "revving", "vroom", "engine revving",
        "tiếng động cơ tăng tốc", "tiếng rồ ga", "rồ ga", "tăng ga",
    ),
    "Clapping": (
        "clapping", "hand claps", "applause", "tiếng vỗ tay", "vỗ tay",
    ),
    "Laughter": (
        "laughter", "laughing", "laugh", "tiếng cười", "cười", "cười đùa",
    ),
    "Giggle": (
        "giggle", "giggling", "tiếng cười khúc khích", "cười khúc khích",
    ),
    "Conversation": (
        "conversation", "people talking", "conversation between people",
        "tiếng trò chuyện", "mọi người trò chuyện", "nói chuyện",
    ),
    "Shout": (
        "shout", "shouting", "yell", "yelling", "tiếng hét", "hét", "la hét",
    ),
    "Crying_and_sobbing": (
        "crying", "sobbing", "cry", "tiếng khóc", "khóc", "nức nở",
    ),
    "Knock": (
        "knock", "knocking", "table knock", "tiếng gõ", "tiếng đập bàn",
        "gõ bàn", "đập bàn",
    ),
    "Slam": (
        "slam", "slamming", "tiếng đóng mạnh", "tiếng đập mạnh",
        "đóng sầm", "đập ghế",
    ),
    "Thump_and_thud": (
        "thump", "thud", "heavy impact", "tiếng va đập trầm", "tiếng thình thịch",
        "tiếng đập ghế", "va đập mạnh",
    ),
    "Traffic_noise_and_roadway_noise": (
        "traffic noise", "roadway noise", "road noise", "tiếng giao thông",
        "tiếng ồn giao thông", "tiếng ồn đường phố", "tiếng xe cộ",
    ),
    "Heavy_engine_(low_frequency)": (
        "heavy engine", "low frequency engine", "truck engine",
        "tiếng động cơ hạng nặng", "tiếng máy xe tải", "tiếng động cơ trầm",
    ),
    "Medium_engine_(mid_frequency)": (
        "medium engine", "mid frequency engine", "vehicle engine",
        "tiếng động cơ xe", "tiếng máy xe", "động cơ tần số trung",
    ),
    "Reversing_beeps": (
        "reversing beeps", "reverse beep", "backup alarm",
        "tiếng bíp lùi xe", "còi lùi xe", "tiếng xe lùi",
    ),
    "Speech": (
        "speech", "speaking", "spoken voice", "tiếng người nói", "lời nói",
        "giọng nói",
    ),
    "Air_horn_and_truck_horn": (
        "air horn", "truck horn", "lorry horn", "còi hơi", "tiếng còi xe tải",
        "còi xe tải",
    ),
    "Police_car_(siren)": (
        "police siren", "police car siren", "tiếng còi xe cảnh sát",
        "còi cảnh sát", "còi hú cảnh sát",
    ),
    "Engine_starting": (
        "engine starting", "engine ignition", "starting a vehicle",
        "tiếng động cơ khởi động", "tiếng khởi động xe", "tiếng đề máy",
    ),
}


@dataclass(frozen=True)
class ParsedAudioQuestion:
    intent: str
    labels: tuple[str, ...] = ()
    ordinals: tuple[int, ...] = ()
    speaker: str | None = None
    quote: str | None = None
    confidence: float = 0.0
    reason: str = "qwen_json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from an LLM continuation."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        depth = 0
        quoted = False
        escaped = False
        for end in range(start, len(stripped)):
            current = stripped[end]
            if quoted:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    quoted = False
                continue
            if current == '"':
                quoted = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : end + 1]
                    try:
                        value = json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            python_literal = re.sub(r"\bnull\b", "None", candidate)
                            python_literal = re.sub(r"\btrue\b", "True", python_literal, flags=re.IGNORECASE)
                            python_literal = re.sub(r"\bfalse\b", "False", python_literal, flags=re.IGNORECASE)
                            value = ast.literal_eval(python_literal)
                        except (SyntaxError, ValueError):
                            break
                    return value if isinstance(value, dict) else None
    # Small schema repair for a common local-model formatting error such as
    # ``"intent":"before", ...}``.  This does not infer semantics from the
    # question; it only restores the missing opening brace.
    if not stripped.startswith("{") and stripped.endswith("}") and '"intent"' in stripped:
        try:
            repaired = json.loads("{" + stripped)
        except json.JSONDecodeError:
            repaired = None
        if isinstance(repaired, dict):
            return repaired
    return None


def _label_aliases(label: str) -> set[str]:
    values = {normalize_text(label)}
    values.add(normalize_text(re.sub(r"\([^)]*\)", " ", label)))
    values.add(normalize_text(label.replace("_and_", " ").replace("_", " ")))
    values.update(normalize_text(alias) for alias in KNOWN_LABEL_ALIASES.get(label, ()))
    return {value for value in values if value}


def _ontology_with_aliases(labels: Sequence[str]) -> str:
    rows: list[str] = []
    for label in labels:
        aliases = KNOWN_LABEL_ALIASES.get(label, ())
        gloss = "; ".join(aliases) if aliases else label.replace("_", " ")
        rows.append(f"- {label} = {gloss}")
    return "\n".join(rows)


def normalize_label(value: Any, labels: Sequence[str]) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"null", "none", "n/a"}:
        return None
    if raw in labels:
        return raw
    normalized = normalize_text(raw)
    matches = [label for label in labels if normalized in _label_aliases(label)]
    if len(matches) == 1:
        return matches[0]
    # Accept a unique ontology alias contained in a longer LLM phrase.
    contained = [
        label
        for label in labels
        if any(
            re.search(rf"(?:^| ){re.escape(alias)}(?: |$)", normalized)
            for alias in _label_aliases(label)
        )
    ]
    return contained[0] if len(contained) == 1 else None


def grounded_labels_in_question(question: str, labels: Sequence[str]) -> tuple[str, ...]:
    """Return ontology labels explicitly grounded by bilingual aliases.

    This is a constrained entity-linking check after LLM parsing.  It neither
    reads the audio inventory nor predicts the answer event.
    """

    normalized_question = f" {normalize_text(question)} "
    hits: list[tuple[int, int, str]] = []
    for label in labels:
        best: tuple[int, int, str] | None = None
        for alias in _label_aliases(label):
            needle = f" {alias} "
            position = normalized_question.find(needle)
            if position < 0:
                continue
            candidate = (position, -len(alias), label)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            hits.append(best)
    hits.sort()
    return tuple(label for _, _, label in hits)


def normalize_program(
    payload: Mapping[str, Any] | None,
    labels: Sequence[str],
) -> ParsedAudioQuestion:
    if payload is None:
        return ParsedAudioQuestion("unsupported", reason="invalid_qwen_json")
    intent = str(payload.get("intent", "unsupported")).strip().lower()
    if intent not in SUPPORTED_INTENTS:
        intent = "unsupported"
    raw_labels = payload.get("labels", [])
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]
    normalized_labels: list[str] = []
    for value in raw_labels if isinstance(raw_labels, list) else []:
        label = normalize_label(value, labels)
        if label is not None and label not in normalized_labels:
            normalized_labels.append(label)
    raw_ordinals = payload.get("ordinals", [])
    if isinstance(raw_ordinals, (int, float, str)):
        raw_ordinals = [raw_ordinals]
    ordinals: list[int] = []
    for value in raw_ordinals if isinstance(raw_ordinals, list) else []:
        try:
            ordinals.append(max(1, int(value)))
        except (TypeError, ValueError):
            ordinals.append(1)
    speaker = str(payload.get("speaker") or "").strip().lower() or None
    if speaker not in {None, "any", "male", "female", "child"}:
        speaker = "any"
    # In a gender-identification question, words such as "male or female" are
    # answer choices, not a requested-speaker filter.
    if intent in {"speaker_identity", "speaker_count"}:
        speaker = "any"
    try:
        confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    quote = str(payload.get("quote") or "").strip() or None
    if intent not in {"speech_before_quote", "speech_after_quote"}:
        quote = None
    return ParsedAudioQuestion(
        intent=intent,
        labels=tuple(normalized_labels),
        ordinals=tuple(ordinals),
        speaker=speaker,
        quote=quote,
        confidence=confidence,
    )


def qwen_parser_prompt(question: str, labels: Sequence[str]) -> str:
    ontology = _ontology_with_aliases(labels)
    return f"""You are the language-understanding component of an AudioQA system.
Convert the user's question into exactly one JSON object. Do not answer the
question and do not use markdown.

Allowed intents:
- list: list every sound/event present
- exists: ask whether a named sound exists
- locate: ask when/where a named sound occurs
- count: ask how many occurrences of a named sound
- first, last, longest: ask for the first/last/longest event
- before, after: ask for the immediately adjacent event around one named anchor
- between: ask what occurs between two named anchors
- overlap: ask what overlaps a named event, or which events overlap
- speaker_count: ask how many distinct people/speakers are speaking
- speaker_identity: ask whether the detected speaker is male, female, or a child
- speech_content: ask what a person/man/woman/child said or request transcription
- open_audioqa: a meaningful audio question outside the operations above
- unsupported: not a question about the audio

Schema:
{{"intent":"...","labels":["exact ontology label"],"ordinals":[1],
 "speaker":"any|male|female|child|null","confidence":0.0}}

Rules:
1. Return ontology labels exactly as written. Use [] when no event label is needed.
2. before/after uses labels[0] as anchor and ordinals[0] as its occurrence.
3. between uses two labels and two ordinals in textual order.
4. exists/locate/count uses labels[0].
5. Do not map phrases such as "at the beginning" to list; use first.
6. Questions such as "what did the woman say?", "người nói nói gì?", or
   "transcribe the speech" are speech_content, not open_audioqa.
7. Questions asking whether the speaker/voice is male or female are
   speaker_identity. They do not ask what was said.
8. Questions asking how many people/speakers are talking are speaker_count,
   not count. They do not require an event label.

Examples:
Question: What sound happens after the second Camera?
{{"intent":"after","labels":["Camera"],"ordinals":[2],"speaker":null,"confidence":1.0}}
Question: What sounds at the end of the audio?
{{"intent":"last","labels":[],"ordinals":[],"speaker":null,"confidence":1.0}}
Question: Người phụ nữ đã nói gì trong đoạn ghi âm?
{{"intent":"speech_content","labels":[],"ordinals":[1],"speaker":"female","confidence":1.0}}
Question: Trong đoạn ghi âm, tiếng người nói là nam hay nữ?
{{"intent":"speaker_identity","labels":[],"ordinals":[],"speaker":"any","confidence":1.0}}
Question: Đoạn ghi âm có bao nhiêu người nói?
{{"intent":"speaker_count","labels":[],"ordinals":[],"speaker":"any","confidence":1.0}}
Question: Is a Meow audible?
{{"intent":"exists","labels":["Meow"],"ordinals":[1],"speaker":null,"confidence":1.0}}

Ontology ({len(labels)} labels; aliases are English and Vietnamese):
{ontology}
Question: {question}
JSON:"""


def qwen_intent_prompt(question: str) -> str:
    """First-stage prompt: classify semantics without a long label ontology."""

    return f"""Classify one AudioQA question. Return exactly one JSON object and no markdown.
Schema: {{"intent":"one value","speaker":"any|male|female|child|null","confidence":0.0}}
Allowed intents: list, exists, locate, count, first, last, longest, before,
after, between, overlap, speaker_identity, speech_content, speech_before_quote,
speech_after_quote, speaker_count, open_audioqa, unsupported.

Definitions:
- list asks for all sounds/events; exists asks yes/no whether one sound is present.
- locate asks when/where/timestamp of one sound; count asks how many times.
- first/last asks which event is at the beginning/end of the audio; longest asks duration.
- before/after asks for the adjacent event before/after one named sound.
- between asks what lies between two named sounds. The word "first" before a
  sound is only its occurrence number; it does NOT mean intent=first or between.
- overlap asks which sound occurs simultaneously with another.
- speaker_count asks for the number of distinct people speaking. It is not the
  number of speech turns and does not require an event label.
- speaker_identity asks whether the detected voice is male, female, or a child.
- speech_content asks what a person said or asks for the whole transcription.
- before/after is about an ENVIRONMENTAL SOUND next to a sound anchor.  The
  phrase "tiếng người nói" or "speech" is a sound anchor, so questions such as
  "Trước tiếng người nói là tiếng gì?" are before/after, NOT speech_content.
- speech_before_quote asks for SPOKEN WORDS before an anchor phrase inside the
  utterance. The anchor phrase may follow the Vietnamese word "câu" and does
  not need quotation marks.
- speech_after_quote asks for SPOKEN WORDS after such an anchor phrase.
- The presence of the word "nói" alone is not enough to choose speech_content.
- A question such as "người nói là nam hay nữ?" is speaker_identity, not
  speech_content and not open_audioqa.
- Speaker is a literal slot: use female only when the question explicitly says
  woman/female/phụ nữ/cô/chị; male only for man/male/đàn ông/anh/chú; child only
  for child/trẻ em. Generic "người nói" or "speaker" MUST be speaker=any.
- open_audioqa is another valid question about audio not covered above.

Examples:
What happens after the first Frying sound?
{{"intent":"after","speaker":null,"confidence":1.0}}
What sounds at the end of the audio?
{{"intent":"last","speaker":null,"confidence":1.0}}
List every event in this recording.
{{"intent":"list","speaker":null,"confidence":1.0}}
When does the Frying sound occur?
{{"intent":"locate","speaker":null,"confidence":1.0}}
Is a cat meowing in the clip?
{{"intent":"exists","speaker":null,"confidence":1.0}}
How many times is the camera heard?
{{"intent":"count","speaker":null,"confidence":1.0}}
What is between Camera and Slam?
{{"intent":"between","speaker":null,"confidence":1.0}}
Người phụ nữ nói gì?
{{"intent":"speech_content","speaker":"female","confidence":1.0}}
Trong đoạn ghi âm trên tiếng người nói là nam hay nữ?
{{"intent":"speaker_identity","speaker":"any","confidence":1.0}}
Đoạn ghi âm có mấy người nói?
{{"intent":"speaker_count","speaker":"any","confidence":1.0}}
How many distinct speakers are in the recording?
{{"intent":"speaker_count","speaker":"any","confidence":1.0}}
Người nói đã nói gì trong đoạn ghi âm?
{{"intent":"speech_content","speaker":"any","confidence":1.0}}
Hãy chép lại lời nói trong audio.
{{"intent":"speech_content","speaker":"any","confidence":1.0}}
Trước tiếng người nói là tiếng gì?
{{"intent":"before","speaker":null,"confidence":1.0}}
Sau tiếng người nói có âm thanh nào?
{{"intent":"after","speaker":null,"confidence":1.0}}
Trước câu "nhập vào làn bên trái" thì người phụ nữ đã nói gì?
{{"intent":"speech_before_quote","speaker":"female","confidence":1.0}}
Trước câu nhập vào làn xe bên trái thì người phụ nữ đó đã nói gì?
{{"intent":"speech_before_quote","speaker":"female","confidence":1.0}}
Trước câu nhập vào làn xe bên trái thì người nói đã nói gì?
{{"intent":"speech_before_quote","speaker":"any","confidence":1.0}}
Sau câu rẽ phải ở ngã tư thì người đàn ông nói gì?
{{"intent":"speech_after_quote","speaker":"male","confidence":1.0}}

Question: {question}
JSON:"""


def qwen_slot_prompt(question: str, intent: str, labels: Sequence[str]) -> str:
    """Second-stage prompt: extract only slots after Qwen has fixed the intent."""

    ontology = _ontology_with_aliases(labels)
    if intent in {"before", "after"}:
        rule = (
            "Return the one anchor sound and its occurrence number. "
            "Vietnamese 'tiếng người nói' maps to the exact ontology label Speech. "
            "Set quote to null."
        )
    elif intent == "between":
        rule = "Return the two anchor sounds in textual order and each occurrence number."
    elif intent in {"exists", "locate", "count"}:
        rule = "Return the one sound class being queried and ordinal 1."
    elif intent == "overlap":
        rule = "Return the named anchor if one exists; otherwise return empty lists."
    elif intent == "speech_content":
        rule = (
            "Return empty labels. If the question explicitly asks for speaker 1/2/3 "
            "(người nói thứ nhất/hai/ba), return that speaker number in ordinals; "
            "otherwise return an empty ordinals list."
        )
    elif intent in {"speech_before_quote", "speech_after_quote"}:
        rule = (
            "Return only the quoted or referenced spoken anchor phrase in quote; "
            "event label lists are empty. In Vietnamese, extract the phrase after "
            "the word 'câu' and before 'thì', even when quotation marks are absent."
        )
    else:
        rule = "No sound slots are required; return empty lists."
    return f"""Extract event slots from an AudioQA question. The intent is already fixed
as {intent}; never change or reinterpret it. Return exactly one JSON object, no markdown.
Schema: {{"labels":["exact ontology label"],"ordinals":[1],"quote":"spoken phrase or null","confidence":0.0}}
{rule}
The words first/second/third before a sound specify that sound's occurrence.
For speech_content, return [] when no speaker ordinal is explicitly stated.
For other intents, when no ordinal is explicitly attached to a sound, use 1.
The relation words before/after (Vietnamese trước/sau) never change the ordinal;
in particular, "sau tiếng người nói" means Speech occurrence 1, not 2.
For two different labels in a between question, the usual output is [1, 1],
not [1, 2]. The labels and ordinals arrays must have the same length.
Map either English or Vietnamese wording through the aliases below, then return
the exact internal label at the start of that ontology row. Never return an
alias as the label and never invent a label.
For example, "Frying (food)" maps to "Frying_(food)" and the complete phrase
"Air horn and truck horn" maps to the single label "Air_horn_and_truck_horn".
Always use the field names labels, ordinals, confidence exactly as shown.

Example for intent=between:
Question: What occurs between Frying (food) and Air horn and truck horn?
{{"labels":["Frying_(food)","Air_horn_and_truck_horn"],"ordinals":[1,1],"confidence":1.0}}

Example for intent=before:
Question: Trước tiếng người nói là tiếng gì?
{{"labels":["Speech"],"ordinals":[1],"quote":null,"confidence":1.0}}

Example for intent=after:
Question: Sau tiếng người nói là tiếng gì?
{{"labels":["Speech"],"ordinals":[1],"quote":null,"confidence":1.0}}

Example for intent=speech_before_quote:
Question: Trước câu nhập vào làn xe bên trái thì người phụ nữ đó đã nói gì?
{{"labels":[],"ordinals":[],"quote":"nhập vào làn xe bên trái","confidence":1.0}}

Example for intent=speech_content:
Question: Người nói thứ hai đã nói gì?
{{"labels":[],"ordinals":[2],"quote":null,"confidence":1.0}}

Ontology ({len(labels)} labels; bilingual aliases):
{ontology}
Question: {question}
JSON:"""


def qwen_speaker_prompt(question: str) -> str:
    """Extract the explicitly mentioned speaker without inferring identity."""

    return f"""Extract only the speaker explicitly named in this question.
Return exactly one JSON object:
{{"speaker":"any|male|female|child","evidence":"exact words from question or null"}}.
Do not infer gender from context, the audio, a quoted sentence, or examples.
Use female only for woman/female/phụ nữ/cô/chị.
Use male only for man/male/đàn ông/anh/chú.
Use child only for child/kid/trẻ em/em bé.
Generic người nói/speaker/người nào đó is always any.
Speaker ordinals such as người nói thứ hai/speaker two describe identity index,
not gender, and are always any.
For female/male/child, evidence MUST be the exact gender-bearing phrase copied
verbatim from the current question. For any, evidence MUST be null.

Question: Người phụ nữ nói gì?
{{"speaker":"female","evidence":"người phụ nữ"}}
Question: Người nói đã nói gì trước câu nhập vào làn bên trái?
{{"speaker":"any","evidence":null}}
Question: What did the speaker say after turn left?
{{"speaker":"any","evidence":null}}
Question: Người nói thứ hai nói gì?
{{"speaker":"any","evidence":null}}
Question: {question}
JSON:"""


def _flatten(inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in inventory:
        occurrences = sorted(
            item.get("occurrences", []), key=lambda row: float(row["start_seconds"])
        )
        for occurrence_index, occurrence in enumerate(occurrences, 1):
            events.append(
                dict(occurrence)
                | {
                    "label": str(item["label"]),
                    "display_label": str(item.get("display_label", item["label"])),
                    "class_score": float(item.get("score", 1.0)),
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


def _occurrence(
    events: Sequence[Mapping[str, Any]], label: str, ordinal: int = 1
) -> Mapping[str, Any] | None:
    selected = [event for event in events if str(event["label"]) == label]
    return selected[ordinal - 1] if 0 < ordinal <= len(selected) else None


def _result(
    intent: str,
    answer: str,
    labels: Sequence[str],
    evidence: Sequence[Mapping[str, Any]],
    reason: str,
) -> AudioQAResult:
    return AudioQAResult(True, intent, answer, tuple(labels), tuple(evidence), reason)


def answer_event_graph_program(
    program: ParsedAudioQuestion,
    inventory: Sequence[Mapping[str, Any]],
) -> AudioQAResult:
    """Execute a Qwen program over predictions only; never inspect gold labels."""

    intent = program.intent
    labels = program.labels
    ordinals = program.ordinals
    events = _flatten(inventory)
    if intent in {
        "unsupported",
        "open_audioqa",
        "speaker_identity",
        "speech_content",
        "speech_before_quote",
        "speech_after_quote",
    }:
        return AudioQAResult(False, intent, "UNSUPPORTED", labels, (), intent)

    if intent == "list":
        unique = list(dict.fromkeys(str(event["display_label"]) for event in events))
        return _result(intent, ", ".join(unique) if unique else "NONE", labels, events, "inventory")
    if intent in {"first", "last", "longest"}:
        if not events:
            return _result(intent, "NONE", labels, (), "empty_inventory")
        if intent == "first":
            answer = events[0]
        elif intent == "last":
            answer = events[-1]
        else:
            answer = max(events, key=lambda x: float(x["end_seconds"]) - float(x["start_seconds"]))
        return _result(intent, str(answer["display_label"]), labels, (answer,), "temporal_reduce")
    if intent in {"exists", "locate", "count"}:
        if not labels:
            return AudioQAResult(False, intent, "UNSUPPORTED", labels, (), "missing_label")
        selected = [event for event in events if str(event["label"]) == labels[0]]
        if intent == "exists":
            answer = "YES" if selected else "NO"
        elif intent == "count":
            answer = str(len(selected))
        else:
            answer = ", ".join(
                f"{float(event['start_seconds']):.2f}–{float(event['end_seconds']):.2f}s"
                for event in selected
            ) or "NONE"
        return _result(intent, answer, labels, selected, "inventory_lookup")
    if intent in {"before", "after"}:
        if not labels:
            return AudioQAResult(False, intent, "UNSUPPORTED", labels, (), "missing_anchor")
        anchor = _occurrence(events, labels[0], ordinals[0] if ordinals else 1)
        if anchor is None:
            return _result(intent, "NONE", labels, (), "anchor_absent")
        position = events.index(anchor)
        answer_index = position - 1 if intent == "before" else position + 1
        if answer_index < 0 or answer_index >= len(events):
            return _result(intent, "NONE", labels, (anchor,), "no_temporal_neighbour")
        answer_event = events[answer_index]
        evidence = sorted((anchor, answer_event), key=lambda x: float(x["start_seconds"]))
        return _result(intent, str(answer_event["display_label"]), labels, evidence, "adjacent_onset")
    if intent == "between":
        if len(labels) < 2:
            return AudioQAResult(False, intent, "UNSUPPORTED", labels, (), "missing_anchors")
        left = _occurrence(events, labels[0], ordinals[0] if ordinals else 1)
        right = _occurrence(events, labels[1], ordinals[1] if len(ordinals) > 1 else 1)
        if left is None or right is None:
            return _result(intent, "NONE", labels, (), "anchor_absent")
        left, right = sorted((left, right), key=lambda x: float(x["start_seconds"]))
        answers = [
            event
            for event in events
            if float(left["start_seconds"]) < float(event["start_seconds"]) < float(right["start_seconds"])
            and event not in (left, right)
        ]
        names = list(dict.fromkeys(str(event["display_label"]) for event in answers))
        return _result(intent, ", ".join(names) if names else "NONE", labels, (left, *answers, right), "between_onsets")
    if intent == "overlap":
        anchors = events
        if labels:
            anchor = _occurrence(events, labels[0], ordinals[0] if ordinals else 1)
            if anchor is None:
                return _result(intent, "NONE", labels, (), "anchor_absent")
            anchors = [anchor]
        evidence: list[Mapping[str, Any]] = []
        answers: list[str] = []
        for anchor in anchors:
            for event in events:
                if event is anchor or str(event["label"]) == str(anchor["label"]):
                    continue
                if min(float(anchor["end_seconds"]), float(event["end_seconds"])) > max(
                    float(anchor["start_seconds"]), float(event["start_seconds"])
                ):
                    for item in (anchor, event):
                        if item not in evidence:
                            evidence.append(item)
                    name = str(event["display_label"])
                    if name not in answers:
                        answers.append(name)
        return _result(intent, ", ".join(answers) if answers else "NONE", labels, evidence, "interval_overlap")
    return AudioQAResult(False, intent, "UNSUPPORTED", labels, (), "unknown_intent")
