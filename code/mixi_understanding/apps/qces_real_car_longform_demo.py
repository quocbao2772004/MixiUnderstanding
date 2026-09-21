#!/usr/bin/env python3
"""Standalone Streamlit page for long-form real-speech car-noise evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "data/qces_real_car_longform_v1"
ASR_DIR = PROJECT_ROOT / "outputs/qces_real_car_longform_asr_v1"
AUDIOSEP_DIR = PROJECT_ROOT / "outputs/qces_real_car_longform_audiosep_v1"
FRCRN_DIR = PROJECT_ROOT / "outputs/qces_real_car_longform_frcrn_v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@st.cache_data(show_spinner=False)
def load_data() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    dataset = _json(DATASET_DIR / "dataset_receipt.json")
    asr_receipt = _json(ASR_DIR / "receipt.json")
    scenes = _jsonl(DATASET_DIR / "scenes.jsonl")
    questions = _jsonl(DATASET_DIR / "questions.jsonl")
    asr_items = {f"{row['scene_id']}::{row['mode']}": row for row in _jsonl(ASR_DIR / "asr_items.jsonl")}
    return dataset, asr_receipt, scenes, questions, asr_items


st.set_page_config(page_title="Real Car Speech Long-form", page_icon="🚗", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1500px; padding-top: 1.3rem; padding-bottom: 3rem;}
    .flow {padding: .8rem 1rem; border: 1px solid #c6d9f2; border-radius: .65rem;
           background: #eef6ff; color: #17385f; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

dataset, asr_receipt, scenes, questions, asr_items = load_data()
test_metrics = asr_receipt["summary"]["test"]

st.title("REAL CAR SPEECH · Long-form noise demo")
st.markdown(
    '<div class="flow"><b>FLEURS real speech</b> + traffic/engine/horn/siren '
    'for 20–30 seconds → AudioSep / FRCRN → Whisper-medium</div>',
    unsafe_allow_html=True,
)
st.warning("Controlled mixture demo, not a recording from a real cabin. Speech is real FLEURS audio; road noise is AudioSet-derived.")

metrics = st.columns(5)
metrics[0].metric("Scenes", dataset["scene_count"])
metrics[1].metric("Duration", f"{min(dataset['duration_seconds'].values()):.1f}-{max(dataset['duration_seconds'].values()):.1f}s")
metrics[2].metric("Mixture WER ↓", f"{test_metrics['mixture']['mean_wer_↓']:.3f}")
metrics[3].metric("FRCRN WER ↓", f"{test_metrics['frcrn']['mean_wer_↓']:.3f}")
metrics[4].metric("AudioSep WER ↓", f"{test_metrics['audiosep']['mean_wer_↓']:.3f}")

table = []
for mode, label in (("mixture", "Mixture"), ("audiosep", "AudioSep"), ("frcrn", "FRCRN"), ("clean", "Clean speech upper bound")):
    value = test_metrics[mode]
    table.append({"Input": label, "Mean WER ↓": value["mean_wer_↓"], "WER ≤ 0.25 ↑": value["accuracy_at_wer_0.25_↑"]})
st.dataframe(table, hide_index=True, use_container_width=True)

options = {"Tất cả": "all", "Easy": "easy", "Medium": "medium", "Hard": "hard"}
difficulty_label = st.radio("MỨC NHIỄU", list(options), horizontal=True)
difficulty = options[difficulty_label]
candidate_scenes = [scene for scene in scenes if scene["split"] == "test" and (difficulty == "all" or scene["difficulty"] == difficulty)]
scene_options = {f"{scene['scene_id']} · {scene['difficulty']} · {scene['duration_seconds']:.1f}s": scene for scene in candidate_scenes}
scene = scene_options[st.selectbox("Chọn scene", list(scene_options))]
scene_questions = [question for question in questions if question["scene_id"] == scene["scene_id"]]
question_options = {
    f"{question['operation']} · {question['question']}": question
    for question in scene_questions
}
question = question_options[st.selectbox("Câu hỏi cần trả lời", list(question_options))]

st.subheader("Scene")
st.markdown(f"**Question:** {question['question']}")
st.markdown(f"**Gold answer:** {question['answer']}")
event_by_id = {event["event_id"]: event for event in scene["events"]}
evidence_events = [event_by_id[event_id] for event_id in question["evidence_event_ids"]]
st.dataframe(
    [
        {
            "Evidence role": event["role"],
            "Label": event["display_name"],
            "Span": f"{event['onset_seconds']:.2f}-{event['offset_seconds']:.2f}s",
        }
        for event in evidence_events
    ],
    hide_index=True,
    use_container_width=True,
)

audio_cols = st.columns(4)
audio_paths = {
    "Mixture": _resolve(str(scene["mixture_path"])),
    "AudioSep": AUDIOSEP_DIR / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
    "FRCRN": FRCRN_DIR / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
    "Clean upper bound": _resolve(str(scene["events"][0]["stem_path"])),
}
for column, (label, path) in zip(audio_cols, audio_paths.items()):
    with column:
        st.markdown(f"**{label}**")
        st.audio(str(path))

rows = []
for mode, label in (("mixture", "Mixture"), ("audiosep", "AudioSep"), ("frcrn", "FRCRN"), ("clean", "Clean upper bound")):
    item = asr_items[f"{scene['scene_id']}::{mode}"]
    rows.append({"Input": label, "WER ↓": item["wer_↓"], "Correct @0.25": "✅" if item["correct_at_wer_0.25"] else "❌", "Whisper output": item["hypothesis"]})
st.subheader("Whisper-medium full-audio result")
if question["operation"] == "speech_transcription":
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("Đây là full-audio ASR, không cắt oracle speech span.")
else:
    st.info("Câu hỏi event đã có reference evidence timeline. Event detector/predicted evidence cho long-form chưa được nối; phần này dùng để nghe và kiểm tra ground-truth evidence trước.")

with st.expander("Nghe reference evidence của câu hỏi"):
    for event in evidence_events:
        st.markdown(f"**{event['role']} · {event['display_name']}**")
        st.audio(str(_resolve(event["stem_path"])))
