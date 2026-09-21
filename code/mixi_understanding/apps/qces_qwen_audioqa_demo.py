#!/usr/bin/env python3
"""One-page Qwen-routed event AudioQA and Vietnamese speech-content demo."""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.apps.qces_audioqa_simple_demo import (  # noqa: E402
    answer_is_correct,
    demo_audit_score,
    evidence_panel,
    load_demo,
    load_old30_labels,
    oracle_inventory,
    wav_bytes,
)
from mixi_understanding.audioqa_qwen_contract import (  # noqa: E402
    ParsedAudioQuestion,
    answer_event_graph_program,
)
from mixi_understanding.audioqa_event_graph import AudioQAResult, normalize_text  # noqa: E402


QWEN_URL = "http://127.0.0.1:8511"
LIVE_INFERENCE_URL = "http://127.0.0.1:8512"
LIVE_AUDIO_ROOT = PROJECT_ROOT / "outputs/qces_live_recordings"
SPEECH_DATASET = PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_realistic_v1"
HONEST_OUTPUT = PROJECT_ROOT / "outputs/qces_combined_honest_zero_shot_v1"
HONEST_PREDICTIONS = HONEST_OUTPUT / "combined_predictions.jsonl"
HONEST_RECEIPT = HONEST_OUTPUT / "receipt.json"
COMBINED_ONTOLOGY = (
    "Accelerating_and_revving",
    "Bark",
    "Howl",
    "Whimper_(dog)",
    "Meow",
    "Purr",
    "Caterwaul",
    "Clapping",
    "Laughter",
    "Giggle",
    "Conversation",
    "Shout",
    "Crying_and_sobbing",
    "Knock",
    "Slam",
    "Thump_and_thud",
    "Traffic_noise_and_roadway_noise",
    "Heavy_engine_(low_frequency)",
    "Medium_engine_(mid_frequency)",
    "Reversing_beeps",
    "Speech",
    "Air_horn_and_truck_horn",
    "Police_car_(siren)",
    "Engine_starting",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def post_qwen(endpoint: str, payload: Mapping[str, Any], timeout: int = 180) -> dict[str, Any]:
    request = urllib.request.Request(
        QWEN_URL + endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "error" in result:
        raise RuntimeError(f"{result['error']}: {result.get('message', '')}")
    return result


def qwen_health() -> bool:
    try:
        with urllib.request.urlopen(QWEN_URL + "/health", timeout=0.8) as response:
            return bool(json.loads(response.read().decode("utf-8")).get("ok"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def live_inference_health() -> bool:
    try:
        with urllib.request.urlopen(LIVE_INFERENCE_URL + "/health", timeout=0.8) as response:
            return bool(json.loads(response.read().decode("utf-8")).get("ok"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def post_live_inference(audio_path: Path, timeout: int = 600) -> dict[str, Any]:
    request = urllib.request.Request(
        LIVE_INFERENCE_URL + "/infer",
        data=json.dumps({"audio_path": str(audio_path.resolve())}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "error" in result:
        raise RuntimeError(f"{result['error']}: {result.get('message', '')}")
    return result


def parsed_program(payload: Mapping[str, Any]) -> ParsedAudioQuestion:
    row = dict(payload["program"])
    # Some Qwen responses put multi-anchor slots in the parsed ``json``
    # envelope while leaving only the first anchor in ``program``. Preserve
    # those slots so temporal comparisons can use both requested events.
    slots = row.get("slots")
    if not isinstance(slots, Mapping):
        parsed_json = payload.get("json")
        slots = parsed_json.get("slots") if isinstance(parsed_json, Mapping) else None
    if isinstance(slots, Mapping):
        for field in ("labels", "ordinals", "quote"):
            current = row.get(field)
            candidate = slots.get(field)
            if candidate and (
                not current
                or (isinstance(current, (list, tuple)) and len(current) < len(candidate))
            ):
                row[field] = candidate
    return ParsedAudioQuestion(
        intent=str(row["intent"]),
        labels=tuple(str(value) for value in row.get("labels", [])),
        ordinals=tuple(int(value) for value in row.get("ordinals", [])),
        speaker=row.get("speaker"),
        quote=row.get("quote"),
        confidence=float(row.get("confidence", 0.0)),
        reason=str(row.get("reason", "qwen_json")),
    )


def compact_program(program: ParsedAudioQuestion) -> str:
    parts = [f"intent={program.intent}"]
    if program.labels:
        parts.append("label=" + ", ".join(label.replace("_", " ") for label in program.labels))
    if program.ordinals:
        parts.append("ordinal=" + ", ".join(str(x) for x in program.ordinals))
    if program.speaker:
        parts.append(f"speaker={program.speaker}")
    if program.quote:
        parts.append(f'quote="{program.quote}"')
    parts.append(f"confidence={program.confidence:.2f}")
    return " · ".join(parts)


def answer_box(label: str, value: str) -> None:
    """Render long answers without the truncation imposed by st.metric."""

    st.markdown(f"**{label}**")
    st.markdown(
        f'<div class="answer-box">{html.escape(str(value))}</div>',
        unsafe_allow_html=True,
    )


def speaker_answer(value: str) -> str:
    """Return a user-facing answer without pretending an unknown prediction is known."""

    return {
        "male": "Nam",
        "female": "Nữ",
        "child": "Trẻ em",
        "multiple": "Có nhiều người nói",
        "unknown": "Không xác định",
    }.get(str(value).lower(), "Không xác định")


def word_error_rate(reference: str, hypothesis: str) -> float:
    left = reference.lower().strip(" .,!?:;\"'").split()
    right = hypothesis.lower().strip(" .,!?:;\"'").split()
    previous = list(range(len(right) + 1))
    for index, source in enumerate(left, 1):
        current = [index]
        for column, target in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (source != target),
                )
            )
        previous = current
    return previous[-1] / max(len(left), 1)


@st.cache_data(show_spinner=False)
def crop_audio(path_text: str, start: float, end: float) -> bytes:
    """Return a browser-friendly WAV crop without loading the whole mixture."""

    duration = max(0.02, float(end) - float(start))
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, float(start)):.6f}",
            "-i",
            path_text,
            "-t",
            f"{duration:.6f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout


def materialize_live_audio(raw: bytes, suffix: str) -> tuple[Path, Path, str]:
    """Persist one browser recording and standardize it for honest inference."""

    digest = hashlib.sha256(raw).hexdigest()[:20]
    run_dir = LIVE_AUDIO_ROOT / digest
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = suffix.lower() if suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"} else ".bin"
    original = run_dir / f"original{safe_suffix}"
    # Keep the complete user recording for playback and future chunked
    # inference.  The current detector explicitly reports when it only
    # analyzes its first 10 seconds; the UI must never silently replace the
    # uploaded original with a truncated waveform.
    standardized = run_dir / "input_16k_mono_full.wav"
    if not original.is_file():
        original.write_bytes(raw)
    if not standardized.is_file():
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(original), "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(standardized),
            ],
            check=True,
            capture_output=True,
        )
    return standardized, run_dir, digest


def group_live_inventory(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        label = str(event["label"])
        item = grouped.setdefault(
            label,
            {
                "label": label,
                "display_label": str(event.get("display_label") or label),
                "score": float(event.get("confidence", 0.0)),
                "occurrences": [],
            },
        )
        item["score"] = max(float(item["score"]), float(event.get("confidence", 0.0)))
        item["occurrences"].append(
            {
                "start_seconds": float(event["start_seconds"]),
                "end_seconds": float(event["end_seconds"]),
            }
        )
    return list(grouped.values())


@st.cache_data(show_spinner=False)
def load_speech_demo() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    scenes = [row for row in read_jsonl(SPEECH_DATASET / "scenes.jsonl") if row["split"] == "test"]
    predictions = {
        str(row["scene_id"]): row
        for row in read_jsonl(HONEST_PREDICTIONS)
        if row["split"] == "test"
    }
    receipt = json.loads(HONEST_RECEIPT.read_text(encoding="utf-8"))
    return scenes, predictions, receipt


def oracle_combined_inventory(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in scene["events"]:
        if event["event_kind"] == "speech":
            label = "Speech"
            display = f"Speech – tiếng người nói ({'nữ' if event['speaker_group'] == 'female' else 'nam'})"
        else:
            label = str(event["label"])
            display = str(event["display_name"])
        item = grouped.setdefault(
            label,
            {"label": label, "display_label": display, "score": 1.0, "occurrences": []},
        )
        item["occurrences"].append(
            {
                "start_seconds": float(event["onset_seconds"]),
                "end_seconds": float(event["offset_seconds"]),
            }
        )
    return list(grouped.values())


def boundary_relation_for_speech(
    program: ParsedAudioQuestion,
    inventory: list[dict[str, Any]],
) -> AudioQAResult:
    """Use speech offset/onset boundaries, not merely adjacent event onsets."""

    if program.intent not in {"before", "after"} or not program.labels or program.labels[0] != "Speech":
        return answer_event_graph_program(program, inventory)
    flat = []
    for item in inventory:
        for occurrence in item["occurrences"]:
            flat.append(
                dict(occurrence)
                | {"label": item["label"], "display_label": item["display_label"]}
            )
    anchors = sorted(
        [event for event in flat if event["label"] == "Speech"],
        key=lambda event: event["start_seconds"],
    )
    ordinal = program.ordinals[0] if program.ordinals else 1
    if ordinal > len(anchors):
        return AudioQAResult(True, program.intent, "NONE", program.labels, (), "speech_anchor_absent")
    anchor = anchors[ordinal - 1]
    candidates = [event for event in flat if event["label"] != "Speech"]
    if program.intent == "before":
        candidates = [event for event in candidates if event["end_seconds"] <= anchor["start_seconds"] + 0.05]
        answer = max(candidates, key=lambda event: event["end_seconds"], default=None)
    else:
        candidates = [event for event in candidates if event["start_seconds"] >= anchor["end_seconds"] - 0.05]
        answer = min(candidates, key=lambda event: event["start_seconds"], default=None)
    if answer is None:
        return AudioQAResult(True, program.intent, "NONE", program.labels, (anchor,), "no_boundary_neighbour")
    evidence = tuple(sorted((anchor, answer), key=lambda event: event["start_seconds"]))
    return AudioQAResult(
        True,
        program.intent,
        str(answer["display_label"]),
        program.labels,
        evidence,
        "non_overlapping_speech_boundary",
    )


def best_quote_span(quote: str, words: list[Mapping[str, Any]]) -> tuple[int, int, float] | None:
    query = normalize_text(quote)
    if not query or not words:
        return None
    target_length = max(1, len(query.split()))
    best: tuple[int, int, float] | None = None
    for start in range(len(words)):
        for length in range(max(1, target_length - 2), min(len(words) - start, target_length + 2) + 1):
            end = start + length
            candidate = " ".join(normalize_text(str(word["text"])) for word in words[start:end])
            score = difflib.SequenceMatcher(None, query, candidate).ratio()
            if best is None or score > best[2]:
                best = (start, end, score)
    return best


def reference_side(reference: str, quote: str, before: bool) -> str:
    tokens = re.findall(r"\w+", reference, flags=re.UNICODE)
    words = [{"text": token} for token in tokens]
    match = best_quote_span(quote, words)
    if match is None:
        return "NONE"
    start, end, _ = match
    selected = tokens[:start] if before else tokens[end:]
    return " ".join(selected) if selected else "NONE"


def event_mode() -> None:
    summary, scenes = load_demo()
    old30_labels = load_old30_labels()
    taxonomy = [str(label) for label in summary["labels"]]
    curated = [
        row
        for row in scenes
        if int(row["audit_only"]["gold_event_count"]) == 4
        and bool(row["audit_only"]["exact_inventory"])
    ]
    curated.sort(key=lambda row: (demo_audit_score(row), str(row["scene_id"])), reverse=True)
    selected = st.selectbox("Chọn audio", [f"Audio {i + 1}" for i in range(len(curated))])
    scene = curated[int(selected.split()[-1]) - 1]
    st.audio(wav_bytes(scene["mixture_path"]), format="audio/wav")
    sorted_events = sorted(scene["gold_events"], key=lambda event: float(event["start_seconds"]))
    old_count = sum(str(event["label"]) in old30_labels for event in sorted_events)
    left, right = st.columns(2)
    left.metric("Bản cũ", f"{old_count}/4 events thuộc 30 classes")
    right.metric("Bản hiện tại", "4/4 events thuộc 188 classes")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "#": index,
                    "Sự kiện": event["display_label"],
                    "Thời gian": f"{float(event['start_seconds']):.2f}–{float(event['end_seconds']):.2f}s",
                }
                for index, event in enumerate(sorted_events, 1)
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    question = st.text_input(
        "Xoá câu mẫu và hỏi bằng cách diễn đạt của m",
        value=f"What sound happens after the first {sorted_events[1]['display_label']}?",
        key=f"event_question_{scene['scene_id']}",
    )
    if not st.button("Hỏi", type="primary", use_container_width=True, key=f"event_ask_{scene['scene_id']}"):
        st.caption("Qwen hiểu câu tự nhiên; executor hỗ trợ quan hệ, đếm, vị trí, tồn tại và overlap.")
        return
    with st.spinner("Qwen đang hiểu câu hỏi…"):
        parsed = post_qwen("/parse", {"question": question, "labels": taxonomy})
    program = parsed_program(parsed)
    st.info("Qwen hiểu: **" + compact_program(program) + "**")
    with st.expander("JSON do Qwen sinh"):
        st.json(parsed["program"])

    if program.intent == "speech_content":
        st.warning("Audio này không có nhánh speech đã được kiểm chứng. Chọn chế độ Tiếng Việt trong xe.")
        return
    if program.intent == "open_audioqa":
        with st.spinner("Qwen2-Audio đang nghe toàn bộ audio…"):
            answer = post_qwen(
                "/audio-answer",
                {"question": question, "audio_path": str(resolve_path(scene["mixture_path"]))},
            )["answer"]
        st.metric("Qwen2-Audio – trả lời trực tiếp", answer)
        st.caption("Câu open-ended chưa có gold/evidence annotation nên không gắn nhãn đúng/sai.")
        return

    predicted = answer_event_graph_program(program, scene["predicted_inventory"])
    oracle = answer_event_graph_program(program, oracle_inventory(scene["gold_events"]))
    if not predicted.supported or not oracle.supported:
        st.error("Qwen đã hiểu intent nhưng thiếu slot bắt buộc hoặc câu nằm ngoài executor hiện tại.")
        return
    correct = answer_is_correct(predicted, oracle)
    ours, gold = st.columns(2)
    ours.metric("Ours – câu trả lời", predicted.answer)
    gold.metric("Đáp án đúng", oracle.answer)
    if correct:
        st.success("✅ ĐÚNG")
    else:
        st.error("❌ SAI")
    predicted_column, oracle_column = st.columns(2)
    with predicted_column:
        evidence_panel("Predicted evidence", predicted, scene["mixture_path"], oracle=False)
    with oracle_column:
        evidence_panel("Oracle evidence", oracle, scene["mixture_path"], oracle=True)


def speech_mode() -> None:
    scenes, predictions, receipt = load_speech_demo()
    choices = [f"Tình huống {index + 1}" for index in range(len(scenes))]
    selected = st.selectbox("Chọn tình huống trong xe", choices)
    scene = scenes[choices.index(selected)]
    prediction = predictions[str(scene["scene_id"])]
    gold_speech = next(event for event in scene["events"] if event["event_kind"] == "speech")
    predicted_events = list(prediction["predicted_events"])
    predicted_inventory = list(prediction["predicted_inventory"])
    predicted_speaker = str(prediction.get("predicted_speaker") or "unknown")
    test_metrics = receipt["summary"]["test"]
    st.info(
        "Inference thật: **public BEATs-Strong → predicted event/speech spans → "
        "PhoWhisper-large trên predicted speech crop**. Không dùng timestamp hoặc event label "
        "của scene để sinh câu trả lời."
    )
    st.audio(wav_bytes(str(resolve_path(scene["mixture_path"]))), format="audio/flac")
    metrics = st.columns(4)
    metrics[0].metric("Predicted events", len(predicted_events))
    metrics[1].metric("Predicted speaker", predicted_speaker)
    metrics[2].metric("Test event F1 ↑", f"{test_metrics['event_f1_↑']:.3f}")
    metrics[3].metric("Test WER ↓", f"{test_metrics['corpus_wer_↓']:.3f}")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model dự đoán": event["display_label"],
                    "Class": event["label"].replace("_", " "),
                    "Thời gian": f"{float(event['start_seconds']):.2f}–{float(event['end_seconds']):.2f}s",
                    "Nguồn": event["source"],
                }
                for event in predicted_events
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Ground truth chỉ để đối chiếu, không đi vào inference"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Gold event": event["display_name"],
                        "Thời gian": f"{float(event['onset_seconds']):.2f}–{float(event['offset_seconds']):.2f}s",
                    }
                    for event in scene["events"]
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    question = st.text_input(
        "Hỏi chung về lời nói hoặc các sound events trong cùng audio",
        value="Người nói đã nói gì trong đoạn ghi âm?",
        key=f"speech_question_{scene['scene_id']}",
    )
    if not st.button("Hỏi", type="primary", use_container_width=True, key=f"speech_ask_{scene['scene_id']}"):
        st.caption(
            "Ví dụ: Có những sự kiện gì? · Trước tiếng người nói là tiếng gì? · "
            "Người nói đã nói gì trước câu “nhập vào làn bên trái”?"
        )
        return
    with st.spinner("Qwen đang hiểu câu hỏi…"):
        parsed = post_qwen("/parse", {"question": question, "labels": list(COMBINED_ONTOLOGY)})
    program = parsed_program(parsed)
    st.info("Qwen hiểu: **" + compact_program(program) + "**")
    with st.expander("JSON do Qwen sinh"):
        st.json(parsed["program"])

    requested = program.speaker or "any"
    speaker_matches = requested == "any" or requested == predicted_speaker

    if program.intent == "speaker_identity":
        predicted_answer = speaker_answer(predicted_speaker)
        gold_speaker = str(gold_speech.get("speaker_group") or "unknown")
        gold_answer = speaker_answer(gold_speaker)
        left, right = st.columns(2)
        with left:
            answer_box("Ours – giới tính giọng nói", predicted_answer)
        with right:
            answer_box("Đáp án đúng", gold_answer)
        if predicted_speaker == gold_speaker:
            st.success("✅ ĐÚNG")
        elif predicted_speaker == "unknown":
            st.warning("🟡 Model phát hiện speech nhưng chưa đủ chắc để xác định giới tính.")
        else:
            st.error("❌ SAI")
        span = prediction.get("speech_span_seconds")
        if span is not None:
            st.audio(
                crop_audio(str(resolve_path(scene["mixture_path"])), float(span[0]), float(span[1])),
                format="audio/wav",
            )
            st.caption(f"Predicted speech evidence: {float(span[0]):.2f}–{float(span[1]):.2f}s")
        return

    if program.intent in {"speech_before_quote", "speech_after_quote"}:
        if not speaker_matches:
            st.metric("Câu trả lời", "NONE")
            if predicted_speaker == "unknown":
                st.warning("Không xác minh được giới tính speaker từ prediction; không dùng gold để ép trả lời.")
            else:
                st.error("Predicted speaker không khớp speaker được hỏi.")
            return
        if not program.quote:
            st.error("Qwen chưa trích được câu anchor khỏi câu hỏi.")
            return
        words = list(prediction["words"])
        match = best_quote_span(program.quote, words)
        if match is None or match[2] < 0.45:
            st.error("Không ground được câu trích dẫn vào transcript của audio.")
            return
        start, end, score = match
        before = program.intent == "speech_before_quote"
        answer_words = words[:start] if before else words[end:]
        answer = " ".join(str(word["text"]).strip() for word in answer_words).strip() or "NONE"
        gold = reference_side(str(gold_speech["transcript"]), program.quote, before)
        wer = word_error_rate(gold, answer) if gold != "NONE" else float(answer != "NONE")
        anchor_words = words[start:end]
        evidence_words = [*answer_words, *anchor_words] if before else [*anchor_words, *answer_words]
        left, right = st.columns(2)
        with left:
            answer_box("Ours – phần lời nói được hỏi", answer)
        with right:
            answer_box("Đáp án theo transcript gốc", gold)
        st.caption(
            f"Quote grounding similarity={score:.3f} · anchor ASR: "
            + " ".join(str(word["text"]).strip() for word in anchor_words)
        )
        if wer == 0.0:
            st.success("✅ Transcript phần được hỏi khớp hoàn toàn")
        elif wer <= 0.25:
            st.warning(f"🟡 Đạt WER≤0.25 nhưng chưa exact · WER={wer:.3f}")
        else:
            st.error(f"❌ Transcript phần được hỏi chưa đạt · WER={wer:.3f}")
        if evidence_words:
            evidence_start = max(0.0, float(evidence_words[0]["start_seconds"]) - 0.08)
            evidence_end = float(evidence_words[-1]["end_seconds"]) + 0.08
            st.markdown("#### Evidence: answer + quoted anchor")
            st.audio(
                crop_audio(str(resolve_path(scene["mixture_path"])), evidence_start, evidence_end),
                format="audio/wav",
            )
            st.caption(f"{evidence_start:.2f}–{evidence_end:.2f}s")
        return

    if program.intent not in {"speech_content", "open_audioqa", "unsupported"}:
        result = boundary_relation_for_speech(program, predicted_inventory)
        oracle_result = boundary_relation_for_speech(program, oracle_combined_inventory(scene))
        if not result.supported:
            st.error("Intent đã hiểu nhưng thiếu event anchor cần thiết.")
            return
        left, right = st.columns(2)
        with left:
            answer_box("Ours – predicted timeline", result.answer)
        with right:
            answer_box("Ground truth answer", oracle_result.answer)
        if normalize_text(result.answer) == normalize_text(oracle_result.answer):
            st.success("✅ ĐÚNG")
        else:
            st.error("❌ SAI")
        if result.evidence_window is not None:
            start, end = result.evidence_window
            st.markdown("#### Temporal evidence")
            st.audio(
                crop_audio(str(resolve_path(scene["mixture_path"])), start, end),
                format="audio/wav",
            )
            st.caption(f"{start:.2f}–{end:.2f}s")
        return

    if program.intent != "speech_content":
        with st.spinner("Qwen2-Audio đang trả lời câu open-ended…"):
            answer = post_qwen(
                "/audio-answer",
                {"question": question, "audio_path": str(resolve_path(scene["mixture_path"]))},
            )["answer"]
        st.metric("Qwen2-Audio – trả lời trực tiếp", answer)
        st.caption("Đường open-ended chưa có temporal evidence verifier.")
        return

    answer = str(prediction["hypothesis"]) if speaker_matches else "NONE"
    gold = str(gold_speech["transcript"]) if speaker_matches else "NONE"
    wer = word_error_rate(gold, answer) if speaker_matches else 0.0
    left, right = st.columns(2)
    with left:
        answer_box("Ours – PhoWhisper-large", answer)
    with right:
        answer_box("Đáp án đúng", gold)
    if not speaker_matches:
        st.success("✅ ĐÚNG · speaker được hỏi không có trong audio")
    elif wer == 0.0:
        st.success("✅ Transcript khớp hoàn toàn · WER=0.000")
    elif wer <= 0.25:
        st.warning(f"🟡 Đạt ngưỡng WER≤0.25, nhưng không phải transcript exact · WER={wer:.3f}")
    else:
        st.error(f"❌ Chưa đạt · WER={wer:.3f}")
    if speaker_matches:
        predicted_column, oracle_column = st.columns(2)
        with predicted_column:
            st.markdown("#### Predicted speech evidence dùng cho ASR")
            span = prediction.get("speech_span_seconds")
            if span is None:
                st.warning("Model không phát hiện speech span.")
            else:
                st.audio(
                    crop_audio(str(resolve_path(scene["mixture_path"])), float(span[0]), float(span[1])),
                    format="audio/wav",
                )
                st.caption(f"BEATs predicted span: {float(span[0]):.2f}–{float(span[1]):.2f}s")
        with oracle_column:
            st.markdown("#### Oracle clean speech")
            st.audio(wav_bytes(str(resolve_path(gold_speech["stem_path"]))), format="audio/flac")
            st.caption(f"{float(gold_speech['onset_seconds']):.2f}–{float(gold_speech['offset_seconds']):.2f}s · chỉ đối chiếu")


def live_mode() -> None:
    st.info(
        "Audio tự thu được chạy thật qua **public BEATs-Strong → predicted timeline/speech span "
        "→ PhoWhisper-large**. Không có annotation hoặc đáp án mẫu cho audio này."
    )
    st.subheader("1. Thu âm hoặc tải audio")
    st.warning(
        "Nếu nút **Bắt đầu thu** không phản hồi, hãy kiểm tra quyền Microphone. "
        "Trình duyệt chặn thu âm khi mở bằng `http://<IP>:8510`; hãy dùng "
        "`http://localhost:8510` qua SSH port forwarding hoặc một domain HTTPS. "
        "Trong lúc chưa có HTTPS, có thể thu bằng điện thoại rồi tải file lên bên dưới."
    )
    recorded = mic_recorder(
        start_prompt="🎙️ Bắt đầu thu",
        stop_prompt="⏹️ Dừng thu",
        just_once=False,
        use_container_width=True,
        format="wav",
        key="qces_live_microphone",
    )
    uploaded = st.file_uploader(
        "Hoặc tải file audio",
        type=["wav", "flac", "mp3", "m4a", "ogg", "webm"],
        key="qces_live_upload",
    )
    raw: bytes | None = None
    suffix = ".wav"
    source = ""
    if uploaded is not None:
        raw = uploaded.getvalue()
        suffix = Path(uploaded.name).suffix or ".wav"
        source = "upload"
    elif recorded is not None and recorded.get("bytes"):
        raw = bytes(recorded["bytes"])
        suffix = "." + str(recorded.get("format") or "wav").lstrip(".")
        source = "microphone"
    if raw is None:
        st.caption(
            "Nên thu tối đa 10 giây. Có thể nói một câu tiếng Việt và tạo thêm "
            "tiếng còi/tiếng xe để hỏi cả speech lẫn sound event."
        )
        return
    st.audio(raw, format=f"audio/{suffix.lstrip('.')}")
    st.caption(f"Nguồn: {source} · {len(raw) / 1024:.1f} KiB · inference dùng tối đa 10 giây đầu")

    st.subheader("2. Đặt câu hỏi")
    question = st.text_input(
        "Hỏi bằng tiếng Việt hoặc tiếng Anh",
        value="Người nói đã nói gì trong đoạn ghi âm?",
        key="qces_live_question",
    )
    st.caption(
        "Ví dụ: Có những âm thanh gì? · Sau tiếng người nói là tiếng gì? · "
        "Tiếng còi xe tải xuất hiện lúc nào? · Người nói đã nói gì?"
    )
    if not st.button("Phân tích audio và trả lời", type="primary", use_container_width=True):
        return
    if not question.strip():
        st.error("Câu hỏi không được để trống.")
        return
    if not live_inference_health():
        st.error("Service BEATs/PhoWhisper ở port 8512 chưa sẵn sàng.")
        return

    try:
        audio_path, run_dir, digest = materialize_live_audio(raw, suffix)
    except (OSError, subprocess.CalledProcessError) as error:
        st.error(f"Không chuẩn hoá được audio: {error}")
        return
    prediction_key = f"qces_live_prediction_v2_{digest}"
    if prediction_key not in st.session_state:
        with st.spinner("BEATs đang phát hiện sự kiện, sau đó PhoWhisper nhận dạng lời nói…"):
            try:
                st.session_state[prediction_key] = post_live_inference(audio_path)
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                st.error(f"Live inference thất bại: {error}")
                return
    prediction = dict(st.session_state[prediction_key])
    prediction_path = run_dir / "prediction.json"
    prediction_path.write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with st.spinner("Qwen đang hiểu câu hỏi…"):
        try:
            parsed = post_qwen(
                "/parse",
                {"question": question.strip(), "labels": list(COMBINED_ONTOLOGY)},
            )
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            st.error(f"Qwen parser thất bại: {error}")
            return
    program = parsed_program(parsed)

    st.subheader("3. Prediction trên audio vừa thu")
    if prediction.get("truncated_to_10_seconds"):
        st.warning("Audio dài hơn 10 giây; bản live hiện chỉ phân tích 10 giây đầu.")
    predicted_events = list(prediction.get("predicted_events", []))
    if predicted_events:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Sự kiện model dự đoán": event["display_label"],
                        "Thời gian": f"{float(event['start_seconds']):.2f}–{float(event['end_seconds']):.2f}s",
                        "Confidence": round(float(event.get("confidence", 0.0)), 3),
                    }
                    for event in predicted_events
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("BEATs không phát hiện được event nào trong ontology automotive hiện tại.")
    st.info("Qwen hiểu: **" + compact_program(program) + "**")
    with st.expander("Prediction receipt · không có ground truth"):
        st.json(
            {
                "audio_sha256_prefix": digest,
                "models": prediction.get("models"),
                "thresholds": prediction.get("thresholds"),
                "annotations_used": prediction.get("annotations_used"),
                "program": parsed.get("program"),
            }
        )

    inventory = group_live_inventory(predicted_events)
    speech_segments = sorted(
        list(prediction.get("speech_segments", [])),
        key=lambda row: float(row.get("start_seconds", 0.0)),
    )
    predicted_speaker = str(prediction.get("predicted_speaker") or "unknown")
    requested_speaker = program.speaker or "any"
    selected_speech_segments = [
        row
        for row in speech_segments
        if requested_speaker == "any"
        or str(row.get("predicted_speaker") or "unknown") == requested_speaker
    ]
    speaker_matches = bool(selected_speech_segments)

    if program.intent == "speaker_identity":
        answer_box("Ours – giới tính giọng nói", speaker_answer(predicted_speaker))
        st.caption("Kết quả lấy từ BEATs trên speech span dự đoán; audio live không có ground truth.")
        if speech_segments:
            for index, segment in enumerate(speech_segments, 1):
                start = float(segment["start_seconds"])
                end = float(segment["end_seconds"])
                st.audio(crop_audio(str(audio_path), start, end), format="audio/wav")
                st.caption(
                    f"Speech evidence {index}: {start:.2f}–{end:.2f}s · "
                    f"speaker={segment.get('predicted_speaker', 'unknown')}"
                )
        else:
            st.warning("Không phát hiện được speech span để xác định giới tính.")
        return
    selected_words = sorted(
        [word for row in selected_speech_segments for word in row.get("words", [])],
        key=lambda row: float(row.get("start_seconds", 0.0)),
    )

    if program.intent in {"speech_before_quote", "speech_after_quote"}:
        if not speaker_matches:
            answer_box("Ours", "NONE")
            st.warning(
                f"Speaker được hỏi là {requested_speaker}, còn detector dự đoán {predicted_speaker}."
            )
            return
        if not program.quote:
            st.error("Qwen chưa trích được câu anchor.")
            return
        words = selected_words
        match = best_quote_span(program.quote, words)
        if match is None or match[2] < 0.45:
            st.error("Không ground được câu trích dẫn vào word timestamps của PhoWhisper.")
            return
        start, end, score = match
        before = program.intent == "speech_before_quote"
        answer_words = words[:start] if before else words[end:]
        answer = " ".join(str(word["text"]).strip() for word in answer_words).strip() or "NONE"
        answer_box("Ours – phần lời nói được hỏi", answer)
        st.caption(f"Quote-grounding similarity={score:.3f} · không có gold transcript")
        evidence_words = [*answer_words, *words[start:end]] if before else [*words[start:end], *answer_words]
        if evidence_words:
            evidence_start = max(0.0, float(evidence_words[0]["start_seconds"]) - 0.08)
            evidence_end = float(evidence_words[-1]["end_seconds"]) + 0.08
            st.audio(crop_audio(str(audio_path), evidence_start, evidence_end), format="audio/wav")
            st.caption(f"Predicted speech evidence: {evidence_start:.2f}–{evidence_end:.2f}s")
        return

    if program.intent == "speech_content":
        answer = (
            " ".join(
                str(row.get("hypothesis") or "").strip()
                for row in selected_speech_segments
                if str(row.get("hypothesis") or "").strip()
            ).strip()
            or "NONE"
        )
        answer_box("Ours – PhoWhisper-large", answer)
        st.caption("Audio tự thu không có đáp án đúng; UI không tự đánh dấu ĐÚNG/SAI.")
        if speaker_matches:
            st.markdown("#### Predicted speech evidence dùng cho ASR")
            for index, segment in enumerate(selected_speech_segments, 1):
                start = float(segment["start_seconds"])
                end = float(segment["end_seconds"])
                speaker = str(segment.get("predicted_speaker") or "unknown")
                st.audio(crop_audio(str(audio_path), start, end), format="audio/wav")
                st.caption(
                    f"Đoạn {index}: {start:.2f}–{end:.2f}s · predicted speaker={speaker}"
                )
        elif not speaker_matches:
            st.warning(
                f"Không có speech segment nào được dự đoán là {requested_speaker}; "
                f"nhãn toàn clip={predicted_speaker}."
            )
        return

    if program.intent in {"open_audioqa", "unsupported"}:
        if program.intent == "unsupported":
            st.error("Qwen xác định đây không phải câu hỏi về audio.")
            return
        with st.spinner("Qwen2-Audio đang trả lời câu open-ended trực tiếp…"):
            answer = post_qwen(
                "/audio-answer",
                {"question": question.strip(), "audio_path": str(audio_path)},
            )["answer"]
        answer_box("Qwen2-Audio direct baseline", str(answer))
        st.warning("Nhánh open-ended này chưa có evidence verifier của ours.")
        return

    result = boundary_relation_for_speech(program, inventory)
    if not result.supported:
        st.error("Đã hiểu intent nhưng không ground được event anchor vào prediction.")
        return
    answer_box("Ours – predicted timeline", result.answer)
    st.caption("Không có ground truth cho audio tự thu; hãy nghe evidence để kiểm chứng.")
    if result.evidence_window is not None:
        evidence_start, evidence_end = result.evidence_window
        st.audio(
            crop_audio(str(audio_path), evidence_start, evidence_end),
            format="audio/wav",
        )
        st.caption(f"Predicted temporal evidence: {evidence_start:.2f}–{evidence_end:.2f}s")


def main() -> None:
    st.set_page_config(page_title="Qwen Evidence-Grounded AudioQA", page_icon="🎧", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1100px; padding-top: 1.2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {background:#f7f8fb; border:1px solid #e3e6ee;
          border-radius:12px; padding:0.65rem 0.85rem;}
        .answer-box {background:#f7f8fb; border:1px solid #e3e6ee;
          border-radius:12px; padding:0.85rem 1rem; min-height:5.2rem;
          white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.55; font-size:1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Evidence-Grounded AudioQA")
    st.caption("Audio → BEATs/PhoWhisper phân tích → Qwen hiểu câu hỏi → evidence kiểm chứng")
    if not qwen_health():
        st.error("Qwen service chưa sẵn sàng. Model có thể vẫn đang load ở tmux qces_qwen_service.")
        st.stop()
    st.success("Qwen2-Audio 7B (4-bit) đang hoạt động", icon="✅")
    if not live_inference_health():
        st.error("Live BEATs/PhoWhisper service chưa sẵn sàng ở port 8512.")
        st.stop()
    st.success("BEATs-Strong + PhoWhisper-large đang hoạt động", icon="✅")
    benchmark_tab, live_tab = st.tabs(
        ["📊 Demo benchmark có đáp án", "🎙️ Thu âm / upload live"]
    )
    with benchmark_tab:
        speech_mode()
    with live_tab:
        live_mode()


if __name__ == "__main__":
    main()
