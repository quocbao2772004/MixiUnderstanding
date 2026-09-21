#!/usr/bin/env python3
"""Streamlit browser for the multi-speaker gender/order diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "data/qces_multispeaker_speech_event_v1"
PIPELINE_DIR = PROJECT_ROOT / "outputs/qces_multispeaker_speech_pipeline_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _speaker_gender(label: str) -> str | None:
    if label == "Speech_male":
        return "male"
    if label == "Speech_female":
        return "female"
    return None


def _center(event: Mapping[str, Any]) -> float:
    return 0.5 * (float(event["onset_seconds"]) + float(event["offset_seconds"]))


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return max(0.0, min(float(left["offset_seconds"]), float(right["offset_seconds"])) - max(float(left["onset_seconds"]), float(right["onset_seconds"])))


def _execute(operation: str, predicted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    speech = [dict(event) for event in predicted if _speaker_gender(str(event["label"]))]
    speech.sort(key=lambda event: (float(event["onset_seconds"]), -float(event["confidence"])))
    sounds = [dict(event) for event in predicted if not _speaker_gender(str(event["label"]))]
    result: dict[str, Any] = {"answer": "No evidence", "selected": [], "anchors": []}
    if operation == "third_speaker_absent":
        result["selected"] = speech
        return result
    if len(speech) < 2:
        return result
    first, second = speech[0], speech[1]
    if operation == "speaker_first":
        result.update(answer=_speaker_gender(first["label"]), selected=[first], anchors=[first, second])
    elif operation == "speaker_second":
        result.update(answer=_speaker_gender(second["label"]), selected=[second], anchors=[first, second])
    elif operation == "event_before_first_speech":
        candidates = [event for event in sounds if _center(event) < float(first["onset_seconds"])]
        if candidates:
            chosen = max(candidates, key=lambda event: (_center(event), float(event["confidence"])))
            result.update(answer=chosen["label"], selected=[chosen], anchors=[first])
    elif operation == "event_between_speakers":
        candidates = [event for event in sounds if _center(event) > float(first["offset_seconds"]) and _center(event) < float(second["onset_seconds"])]
        if candidates:
            chosen = min(candidates, key=lambda event: (_center(event), -float(event["confidence"])))
            result.update(answer=chosen["label"], selected=[chosen], anchors=[first, second])
    elif operation == "event_overlap_second_speech":
        candidates = [(event, _overlap(event, second)) for event in sounds]
        candidates = [(event, value) for event, value in candidates if value > 0]
        if candidates:
            chosen = max(candidates, key=lambda item: (item[1], float(item[0]["confidence"])))[0]
            result.update(answer=chosen["label"], selected=[chosen], anchors=[second])
    elif operation == "event_after_second_speech":
        candidates = [event for event in sounds if _center(event) > float(second["offset_seconds"])]
        if candidates:
            chosen = min(candidates, key=lambda event: (_center(event) - float(second["offset_seconds"]), -float(event["confidence"])))
            result.update(answer=chosen["label"], selected=[chosen], anchors=[second])
    return result


@st.cache_data(show_spinner=False)
def load_data() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    dataset_receipt = json.loads((DATASET_DIR / "dataset_receipt.json").read_text(encoding="utf-8"))
    pipeline_receipt = json.loads((PIPELINE_DIR / "receipt.json").read_text(encoding="utf-8"))
    scenes = _jsonl(DATASET_DIR / "scenes.jsonl")
    questions = _jsonl(DATASET_DIR / "questions.jsonl")
    results = {str(row["question_id"]): row for row in _jsonl(PIPELINE_DIR / "pipeline_results.jsonl")}
    return dataset_receipt, pipeline_receipt, scenes, questions, results


if not globals().get("EMBEDDED_IN_QCES_DEMO", False):
    st.set_page_config(page_title="QCES Multi-speaker Speech", page_icon="🎙️", layout="wide")
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

dataset_receipt, pipeline_receipt, scenes, questions, results = load_data()
scene_by_id = {str(scene["scene_id"]): scene for scene in scenes}
questions_by_scene: dict[str, list[dict[str, Any]]] = {}
for question in questions:
    if question["split"] != "train":
        questions_by_scene.setdefault(str(question["scene_id"]), []).append(question)

st.title("MULTI-SPEAKER DEMO · Male/Female + Speaker Order")
st.markdown(
    '<div class="flow"><b>Mixture</b> → BEATs-Strong gender-aware head → '
    '<b>male/female speech spans</b> → answer + speaker order + evidence</div>',
    unsafe_allow_html=True,
)
st.success("Đây là demo mới trên port 8504: model có nhãn Speech_male và Speech_female, không phải demo Speech cũ.")
st.warning("Diagnostic synthetic speech benchmark. Mỗi difficulty mới có 2 test scenes, chưa phải generalization result.")

DIFFICULTY_OPTIONS = {
    "Tất cả": "all",
    "Easy · ít nhiễu": "easy_overlap",
    "Medium · nhiễu vừa": "medium_overlap",
    "Hard · nhiễu mạnh": "hard_overlap",
}
difficulty_label = st.radio("MỨC NHIỄU", list(DIFFICULTY_OPTIONS), horizontal=True)
difficulty = DIFFICULTY_OPTIONS[difficulty_label]

metrics = pipeline_receipt["metrics"]["test"]
cols = st.columns(6)
cols[0].metric("Test QA", f"{100 * metrics['overall_accuracy_↑']:.1f}%")
cols[1].metric("Gender/order", f"{100 * metrics['speaker_gender_accuracy_↑']:.1f}%")
cols[2].metric("Event answer", f"{100 * metrics['event_answer_accuracy_↑']:.1f}%")
cols[3].metric("No evidence", f"{100 * metrics['no_evidence_accuracy_↑']:.1f}%")
cols[4].metric("Evidence IoU", f"{metrics['mean_evidence_iou_↑']:.3f}")
cols[5].metric("Detector F1", f"{metrics['detector']['event_f1_↑']:.3f}")

st.caption(
    f"{dataset_receipt['scene_count']} scenes · {dataset_receipt['question_count']} questions · "
    f"difficulty={dataset_receipt['difficulty_counts']} · measured overlap SNR="
    f"{dataset_receipt['mean_measured_speech_to_overlap_noise_snr_db']:+.2f} dB"
)

candidate_scenes = [scene for scene in scenes if scene["split"] == "test" and (difficulty == "all" or scene["difficulty"] == difficulty)]
scene_options = {
    f"{scene['scene_id']} · {scene['difficulty']}": str(scene["scene_id"])
    for scene in candidate_scenes
}
if not scene_options:
    st.error("Không có scene cho lựa chọn này.")
    st.stop()
scene_id = scene_options[st.selectbox("Test scene", list(scene_options))]
scene = scene_by_id[scene_id]
scene_questions = questions_by_scene[scene_id]

audio_col, info_col = st.columns([1.0, 1.8])
with audio_col:
    st.subheader("Mixture")
    mixture_path = _resolve(str(scene["mixture_path"]))
    st.audio(str(mixture_path))
    st.caption(f"Duration: {scene['duration_seconds']:.2f}s · {scene['difficulty']}")
with info_col:
    st.subheader("Ground-truth scene")
    rows = []
    for event in sorted(scene["events"], key=lambda item: float(item["onset_seconds"])):
        rows.append({
            "Role": event["role"],
            "Label": event["label"],
            "Speaker": event.get("speaker_group") or "-",
            "Span": f"{event['onset_seconds']:.2f}-{event['offset_seconds']:.2f}s",
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)

question_options = {
    f"{'✅' if results[str(question['question_id'])]['correct'] else '❌'} {question['operation']} · {question['question']}": question
    for question in scene_questions
}
question = question_options[st.selectbox("Question", list(question_options))]
result = results[str(question["question_id"])]
execution = _execute(str(question["operation"]), result["predicted_events"])
st.subheader("Question")
st.markdown(f"**{question['question']}**")
answer_cols = st.columns(3)
answer_cols[0].metric("Gold", str(question["answer"]))
answer_cols[1].metric("Predicted", str(result["predicted_answer"]))
answer_cols[2].metric("Evidence IoU", f"{result['evidence_iou_↑']:.3f}")

st.subheader("Detector inventory")
predicted_rows = [
    {
        "Label": event["label"],
        "Span": f"{event['onset_seconds']:.2f}-{event['offset_seconds']:.2f}s",
        "Confidence": f"{event['confidence']:.3f}",
    }
    for event in result["predicted_events"]
]
st.dataframe(predicted_rows, hide_index=True, use_container_width=True)

evidence_cols = st.columns(2)
with evidence_cols[0]:
    st.markdown("**Predicted evidence**")
    st.audio(str(_resolve(result["predicted_evidence_path"])))
    st.caption("Mixture masked by the predicted male/female and sound-event spans.")
with evidence_cols[1]:
    st.markdown("**Oracle evidence**")
    st.audio(str(_resolve(result["oracle_evidence_path"])))
    st.caption("Ground-truth evidence upper bound for this question.")

with st.expander("Nghe riêng source stems"):
    for event in scene["events"]:
        st.write(f"{event['role']} · {event['label']}")
        st.audio(str(_resolve(str(event["stem_path"]))))
