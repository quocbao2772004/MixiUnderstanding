#!/usr/bin/env python3
"""One-page human listening gate for the QCES clean semantic audit."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_ROOT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    / "v5_atst_clean_stats_ceiling_v1/class_audit_v1"
)
QUEUE_PATH = AUDIT_ROOT / "manual_listening_queue.jsonl"
DECISION_PATH = AUDIT_ROOT / "manual_listening_decisions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decisions() -> dict[str, dict]:
    return {str(row["event_id"]): row for row in read_jsonl(DECISION_PATH)}


def save_decision(row: dict) -> None:
    current = decisions()
    current[str(row["event_id"])] = row
    temporary = DECISION_PATH.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for _, value in sorted(current.items())),
        encoding="utf-8",
    )
    os.replace(temporary, DECISION_PATH)


st.set_page_config(page_title="QCES · Clean data audit", page_icon="🎧", layout="wide")
st.title("QCES · Kiểm tra class và dữ liệu sạch")
st.caption("Mục tiêu: nghe clip Gold bị model nhầm để phân biệt lỗi model với label/crop không rõ. Không quyết định nào tự động đổi ontology hay train set.")

queue = read_jsonl(QUEUE_PATH)
saved = decisions()
if not queue:
    st.error(f"Không tìm thấy listening queue: {QUEUE_PATH}")
    st.stop()

actions = sorted({row["action"] for row in queue})
labels = sorted({row["label"] for row in queue})
left, middle, right = st.columns(3)
with left:
    action = st.selectbox("Nhóm audit", ["Tất cả", *actions])
with middle:
    label = st.selectbox("Class", ["Tất cả", *labels])
with right:
    state = st.selectbox("Trạng thái", ["Chưa nghe", "Đã nghe", "Tất cả"])

filtered = []
for row in queue:
    is_saved = str(row["event_id"]) in saved
    if action != "Tất cả" and row["action"] != action:
        continue
    if label != "Tất cả" and row["label"] != label:
        continue
    if state == "Chưa nghe" and is_saved:
        continue
    if state == "Đã nghe" and not is_saved:
        continue
    filtered.append(row)

counts = Counter(value.get("decision", "") for value in saved.values())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clip cần nghe", len(queue))
c2.metric("Đã đánh dấu", len(saved))
c3.metric("Label/crop ổn", counts.get("label_correct_and_audible", 0))
c4.metric("Cần loại/sửa", counts.get("wrong_label", 0) + counts.get("inaudible_or_bad_crop", 0))

if not filtered:
    st.success("Không còn clip phù hợp bộ lọc này.")
    st.stop()

options = {
    f"{index + 1:03d} · {row['label']} → model: {row['predicted_label']} · {row['scene_id']}": row
    for index, row in enumerate(filtered)
}
selected_name = st.selectbox("Chọn case", list(options))
row = options[selected_name]
previous = saved.get(str(row["event_id"]), {})

st.subheader("Thông tin case")
info_a, info_b, info_c = st.columns(3)
info_a.markdown(f"**Gold label:** `{row['label']}`")
info_b.markdown(f"**Model dự đoán:** `{row['predicted_label']}`")
info_c.markdown(f"**Độ tin cậy:** {row['predicted_confidence']:.3f}")
st.caption(
    f"Nhóm: {row['action']} · span trong scene: {row['onset_seconds']:.2f}–{row['offset_seconds']:.2f}s · "
    f"gold confidence: {row['gold_confidence']:.3f}"
)

audio_a, audio_b = st.columns(2)
with audio_a:
    st.markdown("**Clip component dùng làm evidence**")
    component = Path(row["component_path"])
    if component.is_file():
        st.audio(str(component))
    else:
        st.error(f"Thiếu file: {component}")
with audio_b:
    st.markdown("**Nguồn Gold đầy đủ**")
    source = Path(str(row.get("source_path") or ""))
    if source.is_file():
        st.audio(str(source))
    else:
        st.warning("Không có source đầy đủ.")

decision_values = [
    "label_correct_and_audible",
    "wrong_label",
    "inaudible_or_bad_crop",
    "ambiguous_between_labels",
    "separator_artifact",
]
decision_labels = {
    "label_correct_and_audible": "Label đúng và nghe rõ",
    "wrong_label": "Label sai",
    "inaudible_or_bad_crop": "Không nghe rõ / crop hỏng",
    "ambiguous_between_labels": "Mơ hồ giữa nhiều label",
    "separator_artifact": "Artifact do tách nguồn",
}
default_decision = previous.get("decision", "label_correct_and_audible")
decision = st.radio(
    "Kết luận sau khi nghe",
    decision_values,
    index=decision_values.index(default_decision) if default_decision in decision_values else 0,
    format_func=lambda value: decision_labels[value],
    horizontal=True,
)
note = st.text_area("Ghi chú", value=str(previous.get("note", "")), placeholder="Ví dụ: nghe giống train whistle hơn steam whistle...")
if st.button("Lưu đánh giá", type="primary"):
    save_decision({
        "event_id": row["event_id"],
        "scene_id": row["scene_id"],
        "label": row["label"],
        "predicted_label": row["predicted_label"],
        "component_path": row["component_path"],
        "source_id": row.get("source_id"),
        "decision": decision,
        "note": note.strip(),
    })
    st.success("Đã lưu. Chọn case tiếp theo hoặc lọc 'Chưa nghe'.")
