#!/usr/bin/env python3
"""One-page manual adjudication UI for QCES hierarchy candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1/v5_hierarchy_readiness_v1"
)
MANIFEST = AUDIT_DIR / "listening_manifest.jsonl"
VERDICTS = AUDIT_DIR / "manual_verdicts.jsonl"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def readable(value: str) -> str:
    return value.replace("_and_", " & ").replace("_", " ")


st.set_page_config(page_title="QCES hierarchy audit", layout="wide")
st.title("QCES · nghe và quyết định 11 cặp class dễ nhầm")
st.caption(
    "Các cặp được chọn chỉ từ design-dev: nhầm hai chiều trên clean audio và có cùng parent trong AudioSet. "
    "Chưa cặp nào bị tự động merge."
)

if not MANIFEST.is_file():
    st.error(f"Chưa có listening manifest: {MANIFEST}")
    st.stop()

rows = read_rows(MANIFEST)
pairs: dict[int, list[dict]] = {}
for row in rows:
    pairs.setdefault(int(row["pair_id"]), []).append(row)
pair_id = st.selectbox(
    "Chọn cặp cần nghe",
    sorted(pairs),
    format_func=lambda value: " ↔ ".join(readable(item) for item in pairs[value][0]["pair_labels"]),
)
selected = pairs[int(pair_id)]
left, right = selected[0]["pair_labels"]

st.subheader(f"{readable(left)} ↔ {readable(right)}")
st.info(
    "Nghe `clean component` trước. `Mixture` chỉ dùng để đối chiếu bối cảnh; đoạn cần nghe trong mixture "
    "được ghi bằng timestamp. Control đúng được trộn lẫn để tránh chỉ nghe các ca model sai."
)

for index, row in enumerate(selected, start=1):
    with st.container(border=True):
        st.markdown(
            f"**Ví dụ {index} · {row['kind']}**  \n"
            f"Gold: `{readable(row['gold_label'])}` · Clean prediction: "
            f"`{readable(row['clean_predicted_label'])}` · Mixture prediction: "
            f"`{readable(row['mixture_predicted_label'])}`"
        )
        clean_column, mixture_column = st.columns(2)
        with clean_column:
            st.caption("Clean component")
            st.audio(str(row["component_path"]))
        with mixture_column:
            st.caption(
                f"Mixture · nghe vùng {float(row['onset_seconds']):.2f}–{float(row['offset_seconds']):.2f}s"
            )
            st.audio(str(row["mixture_path"]))

verdict = st.radio(
    "Kết luận cho cả cặp sau khi nghe",
    [
        "keep_distinct_labels_are_audible",
        "merge_as_acoustic_alias",
        "data_mislabeled_or_inaudible",
        "uncertain_need_more_examples",
    ],
    format_func={
        "keep_distinct_labels_are_audible": "Giữ riêng · nghe phân biệt được",
        "merge_as_acoustic_alias": "Gộp alias · về âm học không phân biệt ổn định",
        "data_mislabeled_or_inaudible": "Data sai/khó nghe · sửa nguồn trước",
        "uncertain_need_more_examples": "Chưa chắc · cần thêm ví dụ",
    }.get,
)
note = st.text_area("Ghi chú ngắn", placeholder="Ví dụ: Police siren nghe rõ nhịp khác fire-engine siren...")
if st.button("Lưu verdict", type="primary", use_container_width=True):
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_id": int(pair_id),
        "pair_labels": [left, right],
        "verdict": verdict,
        "note": note.strip(),
    }
    VERDICTS.parent.mkdir(parents=True, exist_ok=True)
    with VERDICTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    st.success(f"Đã lưu verdict cho {readable(left)} ↔ {readable(right)}")

if VERDICTS.is_file():
    verdict_rows = read_rows(VERDICTS)
    latest = {int(row["pair_id"]): row for row in verdict_rows}
    st.divider()
    st.caption(f"Đã có verdict cho {len(latest)}/11 cặp.")
    st.dataframe(
        [
            {
                "Cặp": " ↔ ".join(readable(item) for item in row["pair_labels"]),
                "Verdict": row["verdict"],
                "Ghi chú": row.get("note", ""),
            }
            for _, row in sorted(latest.items())
        ],
        use_container_width=True,
        hide_index=True,
    )
