#!/usr/bin/env python3
"""One-page mentor demo for QCES acoustic evidence reasoning."""

from __future__ import annotations

import io
import json
import re
import sys
import unicodedata
import wave
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.audioqa_event_graph import answer_event_graph_question


DEMO_ROOT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    / "question_conditioned_acoustic_pair_ranker_v2_demo"
)
SUMMARY_PATH = DEMO_ROOT / "summary.json"
CASES_PATH = DEMO_ROOT / "cases.jsonl"
AUDIOQA_ROOT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    / "audioqa_inventory_demo"
)
AUDIOQA_SUMMARY_PATH = AUDIOQA_ROOT / "summary.json"
AUDIOQA_SCENES_PATH = AUDIOQA_ROOT / "scene_inventories.jsonl"

CATEGORY_LABELS = {
    "correct_joint_evidence": "Đúng đáp án + đúng cả anchor/evidence",
    "correct_answer_wrong_anchor": "Đúng answer, anchor chưa khớp",
    "correct_label_wrong_boundary": "Đúng nhãn, boundary chưa khớp",
    "correct_no_evidence": "Đúng no-evidence và đúng anchor",
    "unverified_no_evidence": "Đúng NONE nhưng anchor chưa khớp",
    "wrong_answer": "Sai answer",
    "wrong_abstention": "Abstain sai trên câu answerable",
    "wrong_no_evidence": "False positive trên câu no-evidence",
    "missing_anchor_proposal": "Không có anchor proposal hợp lệ",
}

BEFORE_QUERY_CUES = ("before", "preceding", "prior to", "trước")
AFTER_QUERY_CUES = ("after", "following", "sau")
UNSUPPORTED_ORDINAL_CUES = (
    "first",
    "second",
    "third",
    "fourth",
    "thứ nhất",
    "thứ hai",
    "thứ ba",
    "lần đầu",
    "lần thứ",
)


@st.cache_data(show_spinner=False)
def load_payload() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return summary, rows


@st.cache_data(show_spinner=False)
def load_audioqa_payload() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(AUDIOQA_SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in AUDIOQA_SCENES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return summary, rows


@st.cache_data(show_spinner=False)
def crop_wav(path_string: str, start: float, end: float) -> bytes:
    path = Path(path_string)
    with wave.open(str(path), "rb") as source:
        parameters = source.getparams()
        sample_rate = source.getframerate()
        total_frames = source.getnframes()
        start_sample = max(0, min(total_frames, int(round(start * sample_rate))))
        end_sample = max(
            start_sample + 1,
            min(total_frames, int(round(end * sample_rate))),
        )
        source.setpos(start_sample)
        selected = source.readframes(end_sample - start_sample)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as destination:
        destination.setparams(parameters)
        destination.writeframes(selected)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def wav_bytes(path_string: str) -> bytes:
    return Path(path_string).read_bytes()


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def seconds(value: list[float] | None) -> str:
    if value is None:
        return "—"
    return f"{value[0]:.2f}–{value[1]:.2f}s"


def status_text(row: Mapping[str, Any]) -> str:
    category = str(row["category"])
    if category in {"correct_joint_evidence", "correct_no_evidence"}:
        return "✅ Đúng đầy đủ"
    if category in {
        "correct_answer_wrong_anchor",
        "correct_label_wrong_boundary",
        "unverified_no_evidence",
    }:
        return "⚠️ Đúng một phần"
    return "❌ Sai"


def normalize_query_text(value: str) -> str:
    """Normalize English/Vietnamese surface text for auditable label matching."""

    decomposed = unicodedata.normalize("NFKD", value.replace("_", " "))
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    without_marks = without_marks.lower().replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def parse_direct_question(
    question: str,
    scene_rows: list[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, dict[str, str | None]]:
    """Map a typed before/after question to the equivalent cached model query.

    The v2 checkpoint conditions on ``(anchor class, before/after)``.  Its full
    development predictions are cached by exactly that key, so this lookup is
    numerically identical to rerunning the ranker while keeping the UI light.
    """

    normalized = normalize_query_text(question)
    if not normalized:
        return None, {"reason": "empty_question", "relation": None, "anchor": None}
    if any(cue in normalized for cue in UNSUPPORTED_ORDINAL_CUES):
        return None, {
            "reason": "ordinal_not_supported_by_checkpoint",
            "relation": None,
            "anchor": None,
        }
    before = any(normalize_query_text(cue) in normalized for cue in BEFORE_QUERY_CUES)
    after = any(normalize_query_text(cue) in normalized for cue in AFTER_QUERY_CUES)
    if before == after:
        return None, {
            "reason": "need_exactly_one_before_or_after_relation",
            "relation": None,
            "anchor": None,
        }
    relation = "before" if before else "after"

    candidates: list[tuple[int, str]] = []
    for row in scene_rows:
        label = str(row["anchor_label"])
        alias = normalize_query_text(label)
        if re.search(rf"(?:^| ){re.escape(alias)}(?: |$)", normalized):
            candidates.append((len(alias), label))
    if not candidates:
        return None, {
            "reason": "anchor_label_not_found_in_selected_scene",
            "relation": relation,
            "anchor": None,
        }
    anchor = max(candidates)[1]
    matches = [
        row
        for row in scene_rows
        if row["relation"] == relation and row["anchor_label"] == anchor
    ]
    if not matches:
        return None, {
            "reason": "query_not_cached_for_selected_scene",
            "relation": relation,
            "anchor": anchor,
        }
    return matches[0], {"reason": "ok", "relation": relation, "anchor": anchor}


def render_direct_question(rows: list[dict[str, Any]]) -> None:
    st.subheader("Hỏi trực tiếp trên audio")
    st.caption(
        "Chọn một scene rồi gõ câu before/after bằng tiếng Anh hoặc tiếng Việt. "
        "Kết quả là prediction thật đã cache từ checkpoint v2, không dùng nhãn đáp án "
        "để trả lời. Hiện checkpoint chưa hỗ trợ câu đếm, ordinal hoặc audio upload mới."
    )
    rows_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_scene.setdefault(str(row["scene_id"]), []).append(row)
    scene_ids = sorted(rows_by_scene)
    selected_scene = st.selectbox(
        "Audio scene",
        scene_ids,
        key="direct_scene",
        format_func=lambda scene_id: (
            f"{scene_id} · "
            + ", ".join(
                event["display_label"]
                for event in rows_by_scene[scene_id][0]["scene_events"]
            )
        ),
    )
    scene_rows = rows_by_scene[selected_scene]
    representative = scene_rows[0]
    st.audio(wav_bytes(representative["mixture_path"]), format="audio/wav")
    available_anchors = sorted(
        {str(row["anchor_display_label"]) for row in scene_rows}
    )
    st.caption("Anchor có thể hỏi trong scene: " + " · ".join(available_anchors))
    default_question = str(scene_rows[0]["question"])
    with st.form("direct_question_form"):
        question = st.text_input(
            "Câu hỏi của người dùng",
            value=default_question,
            key=f"direct_question_{selected_scene}",
            help=(
                "Ví dụ: What sound occurs after Microwave oven? hoặc "
                "Âm thanh nào xuất hiện trước Printer?"
            ),
        )
        submitted = st.form_submit_button("Hỏi model", type="primary")
    if not submitted:
        return

    prediction, parsed = parse_direct_question(question, scene_rows)
    if prediction is None:
        reasons = {
            "ordinal_not_supported_by_checkpoint": (
                "Checkpoint v2 chưa nhận ordinal như first/second/lần thứ hai."
            ),
            "need_exactly_one_before_or_after_relation": (
                "Câu hỏi phải chứa đúng một quan hệ before/after (trước/sau)."
            ),
            "anchor_label_not_found_in_selected_scene": (
                "Không nhận ra tên anchor thuộc scene đang chọn."
            ),
            "query_not_cached_for_selected_scene": "Query này chưa có trong cache inference.",
            "empty_question": "Câu hỏi đang trống.",
        }
        st.error(reasons.get(str(parsed["reason"]), "Unsupported question."))
        st.json(parsed)
        return

    st.success(
        "Parsed query: "
        f"relation={parsed['relation']} · anchor={display_label(str(parsed['anchor']))}"
    )
    answer_column, confidence_column = st.columns(2)
    answer_column.metric("Model answer", prediction["predicted_answer_display_label"])
    confidence_column.metric("Confidence margin", f"{prediction['score_margin']:.3f}")
    predicted_window = prediction["predicted_evidence_window_seconds"]
    if predicted_window is None:
        st.warning("Model không tìm được anchor proposal hợp lệ nên không có evidence.")
    else:
        st.markdown(
            f"**Predicted evidence:** {seconds(predicted_window)} · "
            "được cắt trực tiếp từ mixture"
        )
        st.audio(
            crop_wav(
                prediction["mixture_path"],
                predicted_window[0],
                predicted_window[1],
            ),
            format="audio/wav",
        )
    with st.expander("Timeline thật để mentor kiểm tra kết quả", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Event": event["display_label"],
                        "Onset": f"{event['onset_seconds']:.2f}s",
                        "Offset": f"{event['offset_seconds']:.2f}s",
                    }
                    for event in representative["scene_events"]
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


def render_multi_intent_audioqa() -> None:
    st.subheader("Hỏi trực tiếp – Multi-intent AudioQA")
    if not AUDIOQA_SUMMARY_PATH.is_file() or not AUDIOQA_SCENES_PATH.is_file():
        st.info("Predicted event inventory đang được export; refresh trang sau khi hoàn tất.")
        return
    summary, scenes = load_audioqa_payload()
    taxonomy = [str(label) for label in summary["labels"]]
    curated = [
        row
        for row in scenes
        if int(row["audit_only"]["gold_event_count"]) == 4
        and bool(row["audit_only"]["exact_inventory"])
    ]
    subset = st.radio(
        "Tập scene",
        ("Curated 4-event demo", "Tất cả development scenes"),
        horizontal=True,
        help=(
            "Curated chỉ chọn case có predicted top-4 khớp bốn nhãn thật. "
            "Việc trả lời vẫn chỉ sử dụng prediction, không sử dụng gold."
        ),
    )
    available = curated if subset.startswith("Curated") and curated else scenes

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
        predicted_longest = max(predicted, key=lambda event: event[1] - event[0])[2]
        gold_longest = max(gold, key=lambda event: event[1] - event[0])[2]
        return sum(
            (
                len(predicted) == len(gold),
                predicted[0][2] == gold[0][2],
                predicted[-1][2] == gold[-1][2],
                predicted_longest == gold_longest,
            )
        )

    available = sorted(
        available,
        key=lambda row: (
            bool(row["audit_only"]["exact_inventory"]),
            demo_audit_score(row),
            int(row["audit_only"]["correct_labels"]),
            str(row["scene_id"]),
        ),
        reverse=True,
    )
    selected_scene_id = st.selectbox(
        "Audio scene cho AudioQA",
        [str(row["scene_id"]) for row in available],
        key="audioqa_scene",
    )
    scene = next(row for row in available if row["scene_id"] == selected_scene_id)
    inventory = list(scene["predicted_inventory"])
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### Audio mixture")
        st.audio(wav_bytes(scene["mixture_path"]), format="audio/wav")
        st.caption(
            f"Predicted top-{summary['inventory_size']} inventory · "
            f"scoring={summary['selected_variant']} · 188-class detector"
        )
    with right:
        st.markdown("#### Predicted events (model output)")
        predicted_rows = []
        for item in inventory:
            spans = ", ".join(
                f"{float(event['start_seconds']):.2f}–"
                f"{float(event['end_seconds']):.2f}s"
                for event in item["occurrences"]
            )
            predicted_rows.append(
                {
                    "Event": item["display_label"],
                    "Occurrences": len(item["occurrences"]),
                    "Predicted spans": spans,
                    "Score ↑": float(item["score"]),
                }
            )
        st.dataframe(
            pd.DataFrame(predicted_rows).style.format({"Score ↑": "{:.3f}"}),
            hide_index=True,
            use_container_width=True,
        )

    examples = [
        "What sounds are present in this audio?",
        "What sound occurs first?",
        "What sound occurs last?",
        "What sound lasts the longest?",
    ]
    occurrences = sorted(
        [
            dict(occurrence) | {"display_label": item["display_label"]}
            for item in inventory
            for occurrence in item["occurrences"]
        ],
        key=lambda event: (float(event["start_seconds"]), float(event["end_seconds"])),
    )
    if inventory:
        label = str(inventory[0]["display_label"])
        examples.extend(
            (
                f"Is there a {label}?",
                f"Where does {label} occur?",
                f"How many times does {label} occur?",
            )
        )
    if len(occurrences) >= 2:
        examples.extend(
            (
                f"What sound occurs after {occurrences[0]['display_label']}?",
                f"What sound occurs before {occurrences[-1]['display_label']}?",
            )
        )
    overlap_anchor = next(
        (
            left
            for left in occurrences
            for right in occurrences
            if left is not right
            and str(left["display_label"]) != str(right["display_label"])
            and min(float(left["end_seconds"]), float(right["end_seconds"]))
            > max(float(left["start_seconds"]), float(right["start_seconds"]))
        ),
        None,
    )
    if overlap_anchor is not None:
        examples.append(f"What overlaps {overlap_anchor['display_label']}?")
    if len(occurrences) >= 3:
        examples.append(
            f"What occurs between {occurrences[0]['display_label']} and "
            f"{occurrences[-1]['display_label']}?"
        )
    st.caption(
        "Hỗ trợ: list · exists · locate · count · first/last · longest · "
        "before/after · between · overlap. Có thể hỏi bằng tiếng Anh hoặc tiếng Việt."
    )
    with st.form("multi_intent_audioqa_form"):
        example = st.selectbox(
            "Câu hỏi mẫu", examples, key=f"audioqa_example_{selected_scene_id}"
        )
        custom_question = st.text_input(
            "Hoặc tự nhập câu hỏi (để trống nếu dùng câu mẫu)",
            value="",
            key=f"audioqa_custom_question_{selected_scene_id}",
        )
        submitted = st.form_submit_button("Hỏi AudioQA", type="primary")
    if not submitted:
        return
    question = custom_question.strip() or example
    result = answer_event_graph_question(question, inventory, taxonomy)
    if not result.supported:
        st.error(
            "Câu hỏi chưa thuộc các intent demo hỗ trợ. "
            f"intent={result.intent}, reason={result.reason}"
        )
        return
    answer_column, intent_column = st.columns([1.5, 1])
    answer_column.metric("Ours – answer", result.answer)
    intent_column.metric("Parsed intent", result.intent)
    st.caption(
        f"Executor reason: {result.reason}. Answer được suy ra hoàn toàn từ "
        "predicted inventory ở bảng trên."
    )
    evidence_window = result.evidence_window
    if evidence_window is None:
        st.warning(
            "Không có positive evidence. Với câu phủ định, mixture đầy đủ phía trên "
            "là verification region."
        )
    else:
        st.markdown(
            f"**Predicted evidence:** {evidence_window[0]:.2f}–"
            f"{evidence_window[1]:.2f}s"
        )
        st.audio(
            crop_wav(scene["mixture_path"], evidence_window[0], evidence_window[1]),
            format="audio/wav",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Evidence event": event["display_label"],
                        "Start": f"{float(event['start_seconds']):.2f}s",
                        "End": f"{float(event['end_seconds']):.2f}s",
                    }
                    for event in result.evidence
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with st.expander("Oracle timeline – chỉ dùng để mentor audit", expanded=False):
        st.warning("Các nhãn dưới đây không đi vào parser hoặc executor của AudioQA.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Gold event": event["display_label"],
                        "Start": f"{float(event['start_seconds']):.2f}s",
                        "End": f"{float(event['end_seconds']):.2f}s",
                    }
                    for event in scene["gold_events"]
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


def metric_table(summary: Mapping[str, Any]) -> pd.DataFrame:
    v1 = summary["feature_only_v1_best"]["dev"]
    v2 = summary["ranker_best"]["dev"]
    return pd.DataFrame(
        [
            {
                "Method": "Feature-only ranker v1",
                "Answer label ↑": v1["answerable_label_accuracy_↑"],
                "Answer event ↑": v1["answerable_evidence_accuracy_iou030_↑"],
                "Joint anchor+answer ↑": np.nan,
                "Verified no-evidence ↑": v1["verified_no_evidence_accuracy_↑"],
            },
            {
                "Method": "Acoustic pair ranker v2 (current)",
                "Answer label ↑": v2["answerable_label_accuracy_↑"],
                "Answer event ↑": v2["answerable_answer_event_iou030_accuracy_↑"],
                "Joint anchor+answer ↑": v2[
                    "answerable_joint_evidence_iou030_accuracy_↑"
                ],
                "Verified no-evidence ↑": v2[
                    "verified_no_evidence_iou030_accuracy_↑"
                ],
            },
        ]
    )


def main() -> None:
    st.set_page_config(
        page_title="QCES Evidence Reasoning",
        page_icon="🎧",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {background:#f6f7fb; border:1px solid #e5e7eb;
          border-radius:12px; padding:0.7rem 0.9rem;}
        .small-note {color:#5f6673; font-size:0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not SUMMARY_PATH.is_file() or not CASES_PATH.is_file():
        st.error("Chưa export demo cases. Hãy chạy exporter trước.")
        st.stop()
    summary, rows = load_payload()
    v2 = summary["ranker_best"]["dev"]

    st.title("Question-Conditioned Acoustic Evidence Reasoning")
    st.caption(
        "Mixture → event proposals → question-conditioned acoustic pair ranker → "
        "anchor + answer evidence / verified no-evidence"
    )
    first, second, third, fourth, fifth = st.columns(5)
    first.metric("Ontology", f"{summary['classes']} classes")
    second.metric("Development", f"{summary['scenes']} scenes")
    third.metric("Questions", f"{summary['cases']:,}")
    fourth.metric(
        "Anchor IoU ≥ 0.30 ↑",
        percent(v2["answerable_anchor_iou030_accuracy_↑"]),
    )
    fifth.metric(
        "Verified no-evidence ↑",
        percent(v2["verified_no_evidence_iou030_accuracy_↑"]),
    )

    with st.expander("Kết quả benchmark và cách đọc", expanded=True):
        table = metric_table(summary)
        st.dataframe(
            table.style.format(
                {
                    "Answer label ↑": "{:.3f}",
                    "Answer event ↑": "{:.3f}",
                    "Joint anchor+answer ↑": "{:.3f}",
                    "Verified no-evidence ↑": "{:.3f}",
                },
                na_rep="—",
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown(
            """
            <div class="small-note">
            ↑ càng cao càng tốt. V1 chưa đo joint anchor+answer nên ô đó để trống.
            V2 cải thiện verified no-evidence nhưng chưa cải thiện answer accuracy;
            đây là bottleneck hiện tại cần báo cáo thẳng. Candidate Top-120 vẫn có
            joint ceiling khoảng 80%, nên lỗi chính nằm ở contextual selection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_multi_intent_audioqa()
    st.divider()
    st.subheader("Nghe và kiểm tra từng câu hỏi")
    available_categories = [
        category for category in CATEGORY_LABELS if any(row["category"] == category for row in rows)
    ]
    left_filter, right_filter = st.columns([1, 2])
    with left_filter:
        chosen_category = st.selectbox(
            "Loại case",
            available_categories,
            format_func=lambda value: (
                f"{CATEGORY_LABELS[value]} "
                f"({summary['category_counts'].get(value, 0)})"
            ),
        )
    selected_rows = [row for row in rows if row["category"] == chosen_category]
    selected_rows.sort(
        key=lambda row: (float(row["score_margin"]), str(row["scene_id"])),
        reverse=True,
    )
    with right_filter:
        selected_id = st.selectbox(
            "Câu hỏi",
            [row["record_id"] for row in selected_rows],
            format_func=lambda record_id: next(
                f"{row['question']}  |  gold={row['gold_answer_display_label']}  "
                f"|  pred={row['predicted_answer_display_label']}"
                for row in selected_rows
                if row["record_id"] == record_id
            ),
        )
    row = next(item for item in selected_rows if item["record_id"] == selected_id)

    scene_column, qa_column = st.columns([1.15, 1])
    with scene_column:
        st.markdown("#### 1. Audio mixture và các sự kiện thật")
        st.audio(wav_bytes(row["mixture_path"]), format="audio/wav")
        timeline = []
        for index, event in enumerate(row["scene_events"], 1):
            role = []
            if event["label"] == row["anchor_label"]:
                role.append("anchor")
            if event["label"] == row["gold_answer_label"]:
                role.append("gold answer")
            timeline.append(
                {
                    "#": index,
                    "Event": event["display_label"],
                    "Onset": f"{event['onset_seconds']:.2f}s",
                    "Offset": f"{event['offset_seconds']:.2f}s",
                    "Role": ", ".join(role) or "distractor",
                }
            )
        st.dataframe(pd.DataFrame(timeline), hide_index=True, use_container_width=True)

    with qa_column:
        st.markdown("#### 2. Câu hỏi và quyết định của model")
        st.info(row["question"])
        answer_left, answer_right = st.columns(2)
        answer_left.metric("Gold answer", row["gold_answer_display_label"])
        answer_right.metric("Model answer", row["predicted_answer_display_label"])
        st.markdown(f"### {status_text(row)}")
        detail = pd.DataFrame(
            [
                {
                    "Evidence": "Anchor",
                    "Predicted": seconds(row["predicted_anchor_seconds"]),
                    "Oracle": seconds(row["gold_anchor_seconds"]),
                    "IoU ↑": row["anchor_iou"],
                },
                {
                    "Evidence": "Answer",
                    "Predicted": seconds(row["predicted_answer_seconds"]),
                    "Oracle": seconds(row["gold_answer_seconds"]),
                    "IoU ↑": row["answer_iou"],
                },
            ]
        )
        st.dataframe(
            detail.style.format({"IoU ↑": "{:.3f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Candidate supported: {'yes' if row['candidate_supported'] else 'no'} · "
            f"confidence margin: {row['score_margin']:.3f} · "
            f"category: {row['category']}"
        )

    st.markdown("#### 3. Predicted evidence so với oracle evidence")
    st.warning(
        "Các audio dưới đây là temporal evidence crop từ mixture, chưa qua "
        "AudioSep refinement. Vì scene polyphonic nên vẫn có thể nghe thấy distractor."
    )
    predicted_column, oracle_column = st.columns(2)
    with predicted_column:
        predicted_window = row["predicted_evidence_window_seconds"]
        st.markdown(f"**Predicted evidence window:** {seconds(predicted_window)}")
        if predicted_window is None:
            st.caption("Model không tìm được anchor proposal hợp lệ; không phát audio 0 giây.")
        else:
            st.audio(
                crop_wav(row["mixture_path"], predicted_window[0], predicted_window[1]),
                format="audio/wav",
            )
    with oracle_column:
        oracle_window = row["oracle_evidence_window_seconds"]
        st.markdown(f"**Oracle evidence window:** {seconds(oracle_window)}")
        st.audio(
            crop_wav(row["mixture_path"], oracle_window[0], oracle_window[1]),
            format="audio/wav",
        )

    st.markdown("#### 4. Isolated gold components để kiểm tra bằng tai")
    anchor_audio, answer_audio = st.columns(2)
    with anchor_audio:
        st.markdown(f"**Gold anchor:** {row['anchor_display_label']}")
        anchor_path = row.get("gold_anchor_component_path")
        if anchor_path and Path(anchor_path).is_file():
            st.audio(wav_bytes(anchor_path), format="audio/wav")
        else:
            st.caption("Không có component audio.")
    with answer_audio:
        st.markdown(f"**Gold answer:** {row['gold_answer_display_label']}")
        answer_path = row.get("gold_answer_component_path")
        if answer_path and Path(answer_path).is_file():
            st.audio(wav_bytes(answer_path), format="audio/wav")
        elif row["no_evidence"]:
            st.caption("No-evidence: không có answer component.")
        else:
            st.caption("Không có component audio.")

    st.divider()
    st.caption(
        "Thông điệp báo cáo: proposal/localization đã có ceiling cao, nhưng chọn đúng "
        "answer trong scene 188-class polyphonic vẫn là bottleneck. Demo cho phép kiểm "
        "chứng cả success và failure thay vì chỉ xem một accuracy tổng hợp."
    )


if __name__ == "__main__":
    main()
