#!/usr/bin/env python3
"""Minimal one-page AudioQA demo for a mentor presentation."""

from __future__ import annotations

import io
import json
import sys
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.audioqa_event_graph import (  # noqa: E402
    AudioQAResult,
    answer_event_graph_question,
    normalize_text,
)


DATA_ROOT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    / "audioqa_inventory_demo"
)
SUMMARY_PATH = DATA_ROOT / "summary.json"
SCENES_PATH = DATA_ROOT / "scene_inventories.jsonl"
OLD30_REPORT_PATH = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_30class/full801_val102_e12"
    / "training_report.json"
)


@st.cache_data(show_spinner=False)
def load_demo() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    scenes = [
        json.loads(line)
        for line in SCENES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return summary, scenes


@st.cache_data(show_spinner=False)
def load_old30_labels() -> set[str]:
    if not OLD30_REPORT_PATH.is_file():
        return set()
    payload = json.loads(OLD30_REPORT_PATH.read_text(encoding="utf-8"))
    return {str(label) for label in payload.get("labels", [])}


@st.cache_data(show_spinner=False)
def wav_bytes(path_string: str) -> bytes:
    return Path(path_string).read_bytes()


@st.cache_data(show_spinner=False)
def crop_wav(path_string: str, start: float, end: float) -> bytes:
    with wave.open(path_string, "rb") as source:
        parameters = source.getparams()
        rate = source.getframerate()
        total = source.getnframes()
        left = max(0, min(total, int(round(start * rate))))
        right = max(left + 1, min(total, int(round(end * rate))))
        source.setpos(left)
        selected = source.readframes(right - left)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as destination:
        destination.setparams(parameters)
        destination.writeframes(selected)
    return buffer.getvalue()


def oracle_inventory(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    display: dict[str, str] = {}
    for event in events:
        label = str(event["label"])
        display[label] = str(event["display_label"])
        grouped.setdefault(label, []).append(
            {
                "start_seconds": float(event["start_seconds"]),
                "end_seconds": float(event["end_seconds"]),
            }
        )
    return [
        {
            "label": label,
            "display_label": display[label],
            "score": 1.0,
            "occurrences": sorted(
                occurrences, key=lambda event: float(event["start_seconds"])
            ),
        }
        for label, occurrences in grouped.items()
    ]


def interval_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    intersection = max(
        0.0,
        min(float(left["end_seconds"]), float(right["end_seconds"]))
        - max(float(left["start_seconds"]), float(right["start_seconds"])),
    )
    union = (
        float(left["end_seconds"])
        - float(left["start_seconds"])
        + float(right["end_seconds"])
        - float(right["start_seconds"])
        - intersection
    )
    return intersection / max(union, 1e-8)


def _answer_set(answer: str) -> set[str]:
    return {
        normalize_text(value)
        for value in answer.split(",")
        if normalize_text(value)
    }


def answer_is_correct(predicted: AudioQAResult, oracle: AudioQAResult) -> bool:
    if not predicted.supported or not oracle.supported or predicted.intent != oracle.intent:
        return False
    if predicted.intent in {"list", "between", "overlap"}:
        return _answer_set(predicted.answer) == _answer_set(oracle.answer)
    if predicted.intent == "locate":
        if len(predicted.evidence) != len(oracle.evidence):
            return False
        return all(
            any(
                normalize_text(str(prediction["label"]))
                == normalize_text(str(target["label"]))
                and interval_iou(prediction, target) >= 0.30
                for prediction in predicted.evidence
            )
            for target in oracle.evidence
        )
    return normalize_text(predicted.answer) == normalize_text(oracle.answer)


def demo_audit_score(row: Mapping[str, Any]) -> int:
    predicted = sorted(
        [
            (
                float(occurrence["start_seconds"]),
                float(occurrence["end_seconds"]),
                str(item["label"]),
            )
            for item in row["predicted_inventory"]
            for occurrence in item["occurrences"]
        ]
    )
    gold = sorted(
        (
            float(event["start_seconds"]),
            float(event["end_seconds"]),
            str(event["label"]),
        )
        for event in row["gold_events"]
    )
    if not predicted or not gold:
        return 0
    return sum(
        (
            len(predicted) == len(gold),
            predicted[0][2] == gold[0][2],
            predicted[-1][2] == gold[-1][2],
            max(predicted, key=lambda event: event[1] - event[0])[2]
            == max(gold, key=lambda event: event[1] - event[0])[2],
        )
    )


def evidence_panel(
    title: str,
    result: AudioQAResult,
    mixture_path: str,
    *,
    oracle: bool,
) -> None:
    st.markdown(f"#### {title}")
    window = result.evidence_window
    if window is None:
        st.caption("Không có positive evidence cho câu trả lời này.")
        return
    st.caption(f"{window[0]:.2f}–{window[1]:.2f}s")
    st.audio(crop_wav(mixture_path, window[0], window[1]), format="audio/wav")
    labels = list(
        dict.fromkeys(str(event["display_label"]) for event in result.evidence)
    )
    st.caption(("Gold events: " if oracle else "Predicted events: ") + ", ".join(labels))


def main() -> None:
    st.set_page_config(page_title="Evidence-Grounded AudioQA", page_icon="🎧", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1050px; padding-top: 1.2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {background:#f7f8fb; border:1px solid #e3e6ee;
          border-radius:12px; padding:0.65rem 0.85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not SUMMARY_PATH.is_file() or not SCENES_PATH.is_file():
        st.error("Chưa có predicted inventory để chạy demo.")
        st.stop()
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

    st.title("Evidence-Grounded AudioQA Demo")
    st.caption("Nghe audio → hỏi → xem đáp án và kiểm tra predicted/oracle evidence")

    audio_options = [f"Audio {index + 1}" for index in range(len(curated))]
    selected_audio = st.selectbox("Chọn audio", audio_options)
    scene = curated[audio_options.index(selected_audio)]

    st.markdown("### 1. Audio và các sự kiện")
    st.audio(wav_bytes(scene["mixture_path"]), format="audio/wav")
    sorted_events = sorted(
        scene["gold_events"], key=lambda event: float(event["start_seconds"])
    )
    old_supported = sum(str(event["label"]) in old30_labels for event in sorted_events)
    coverage_old, coverage_current = st.columns(2)
    coverage_old.metric("Bản cũ – 30 classes", f"{old_supported}/4 events được hỗ trợ")
    coverage_current.metric("Bản hiện tại – 188 classes", "4/4 events được hỗ trợ")
    event_rows = [
        {
            "#": index,
            "Sự kiện": event["display_label"],
            "Thời gian": (
                f"{float(event['start_seconds']):.2f}–"
                f"{float(event['end_seconds']):.2f}s"
            ),
            "So với bản 30-class": (
                "Đã có" if str(event["label"]) in old30_labels else "Class mới"
            ),
        }
        for index, event in enumerate(
            sorted_events,
            1,
        )
    ]
    st.dataframe(pd.DataFrame(event_rows), hide_index=True, use_container_width=True)

    st.markdown("### 2. Đặt câu hỏi")
    question = st.text_input(
        "Có thể xoá câu mẫu và nhập câu hỏi của m",
        value="What sounds are present in this audio?",
        key=f"question_{scene['scene_id']}",
    )
    submitted = st.button(
        "Hỏi",
        type="primary",
        use_container_width=True,
        key=f"ask_{scene['scene_id']}",
    )
    if not submitted:
        st.caption(
            "Hỗ trợ: list, exists, locate, count, first/last, longest, "
            "before/after, between và overlap."
        )
        return

    predicted = answer_event_graph_question(
        question, scene["predicted_inventory"], taxonomy
    )
    oracle = answer_event_graph_question(
        question, oracle_inventory(scene["gold_events"]), taxonomy
    )
    if not predicted.supported or not oracle.supported:
        st.error(
            "Câu hỏi chưa được hỗ trợ. "
            f"Parser: intent={predicted.intent}, reason={predicted.reason}."
        )
        return
    correct = answer_is_correct(predicted, oracle)

    st.markdown("### 3. Kết quả")
    st.info(f"Câu backend vừa xử lý: **{question}**")
    ours_column, oracle_column = st.columns(2)
    ours_column.metric("Ours – câu trả lời", predicted.answer)
    oracle_column.metric("Đáp án đúng", oracle.answer)
    if correct:
        st.success(f"✅ ĐÚNG · parsed intent: {predicted.intent}")
    else:
        st.error(f"❌ SAI · parsed intent: {predicted.intent}")

    predicted_column, oracle_evidence_column = st.columns(2)
    with predicted_column:
        evidence_panel(
            "Predicted evidence",
            predicted,
            scene["mixture_path"],
            oracle=False,
        )
    with oracle_evidence_column:
        evidence_panel(
            "Oracle evidence",
            oracle,
            scene["mixture_path"],
            oracle=True,
        )


if __name__ == "__main__":
    main()
