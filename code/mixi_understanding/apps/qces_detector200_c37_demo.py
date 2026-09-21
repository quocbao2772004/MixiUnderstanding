#!/usr/bin/env python3
"""One-page Streamlit demo for the 37-chunk 200-class warm-start detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from mixi_understanding.apps.qces_detector200_demo import (
    event_df,
    event_name,
    format_span,
    interval_iou,
    label_text,
    latest_metrics,
    read_json,
    read_jsonl,
    show_timeline,
    timeline_df,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full200/c37_single_event_headonly_e8_v1"
)
DATA_DIR = PROJECT_ROOT / "outputs/qces_full200_single_event_pretrain_c37_v1"
SNAPSHOT_DIR = PROJECT_ROOT / "outputs/qces_full200_snapshot_37chunks_20260811_v1"


def top_prediction(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None
    return max(events, key=lambda event: float(event.get("confidence", 0.0)))


def case_summary(
    predictions: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        scene_id = str(prediction["scene_id"])
        gold_events = list(prediction.get("gold_events") or [])
        predicted_events = list(prediction.get("predicted_events") or [])
        gold = gold_events[0] if gold_events else {}
        top = top_prediction(predicted_events)
        top_correct = bool(top) and str(top.get("label")) == str(gold.get("label"))
        gold_detected = any(
            str(event.get("label")) == str(gold.get("label"))
            for event in predicted_events
        )
        best_iou = max(
            (
                interval_iou(gold, event)
                for event in predicted_events
                if str(event.get("label")) == str(gold.get("label"))
            ),
            default=0.0,
        )
        source = manifest.get(scene_id, {})
        rows.append(
            {
                "index": index,
                "scene_id": scene_id,
                "gold_label": str(gold.get("label") or ""),
                "gold_name": event_name(gold),
                "ours_label": str((top or {}).get("label") or "no_evidence"),
                "ours_name": event_name(top or {}) or "No evidence",
                "confidence": float((top or {}).get("confidence", 0.0)),
                "top1_correct": top_correct,
                "gold_detected": gold_detected,
                "best_same_label_iou": best_iou,
                "predicted_events": len(predicted_events),
                "audio_path": str(source.get("mixture_path") or ""),
                "duration_seconds": float(source.get("duration_seconds") or 0.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="QCES c37 detector demo", layout="wide")
    st.title("QCES — 200-class detector warm-start demo")
    st.caption(
        "Input audio → frozen BEATs-Strong → learned 200-class temporal head → "
        "answer label + predicted evidence span."
    )
    st.warning(
        "Đây là checkpoint warm-start từ 37 chunk và dev audio chỉ có một event sạch. "
        "Nó kiểm tra nhận dạng + timestamp; chưa phải kết quả QA multi-event cuối."
    )

    report_path = RUN_DIR / "training_report.json"
    predictions_path = RUN_DIR / "val_predictions.jsonl"
    manifest_path = DATA_DIR / "detector_scene_manifest_dev.jsonl"
    data_receipt_path = DATA_DIR / "pretrain_data_receipt.json"
    snapshot_receipt_path = SNAPSHOT_DIR / "snapshot_receipt.json"
    required = [
        report_path,
        predictions_path,
        manifest_path,
        data_receipt_path,
        snapshot_receipt_path,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        st.error("Thiếu artifact:\n" + "\n".join(str(path) for path in missing))
        st.stop()

    report = read_json(str(report_path))
    predictions = read_jsonl(str(predictions_path))
    manifest_rows = read_jsonl(str(manifest_path))
    manifest = {str(row["scene_id"]): row for row in manifest_rows}
    data_receipt = read_json(str(data_receipt_path))
    snapshot_receipt = read_json(str(snapshot_receipt_path))
    cases = case_summary(predictions, manifest)
    metrics = latest_metrics(report)

    with st.expander("Dataset và benchmark hiện tại", expanded=True):
        cols = st.columns(7)
        cols[0].metric("Classes", len(report.get("labels") or []))
        cols[1].metric("Completed chunks", snapshot_receipt["completed_chunks"])
        cols[2].metric("Accepted sources", f"{snapshot_receipt['accepted_items']:,}")
        cols[3].metric("Train scenes", f"{data_receipt['train']['scenes']:,}")
        cols[4].metric("Dev scenes", f"{data_receipt['dev']['scenes']:,}")
        cols[5].metric("Scene F1 ↑", f"{metrics['Scene-label F1 ↑']:.3f}")
        cols[6].metric("Event F1 ↑", f"{metrics['Event F1 @ IoU0.30 ↑']:.3f}")
        st.caption(
            f"Frame F1 ↑ {metrics['Frame F1 ↑']:.3f} · "
            f"Scene recall ↑ {metrics['Scene-label recall ↑']:.3f} · "
            f"Avg false-positive labels ↓ {metrics['Avg FP labels ↓']:.3f}. "
            "↑ càng cao càng tốt; ↓ càng thấp càng tốt."
        )

    st.sidebar.header("Chọn case")
    status = st.sidebar.selectbox(
        "Kết quả",
        ["Tất cả", "Top-1 đúng", "Top-1 sai", "Bỏ sót gold", "IoU < 0.30"],
        index=1,
    )
    query = st.sidebar.text_input("Tìm label", "")
    minimum_duration = st.sidebar.slider(
        "Thời lượng audio tối thiểu (giây)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Player hiển thị clip dưới 1 giây thành 0:00 dù file không rỗng.",
    )
    order = st.sidebar.selectbox(
        "Sắp xếp",
        ["Confidence thấp trước", "Confidence cao trước", "IoU thấp trước", "Theo label"],
        index=1,
    )
    filtered = cases[cases["duration_seconds"] >= minimum_duration].copy()
    if status == "Top-1 đúng":
        filtered = filtered[filtered["top1_correct"]]
    elif status == "Top-1 sai":
        filtered = filtered[~filtered["top1_correct"]]
    elif status == "Bỏ sót gold":
        filtered = filtered[~filtered["gold_detected"]]
    elif status == "IoU < 0.30":
        filtered = filtered[filtered["best_same_label_iou"] < 0.30]
    if query.strip():
        needle = query.strip().lower()
        filtered = filtered[
            filtered["gold_name"].str.lower().str.contains(needle, regex=False)
            | filtered["ours_name"].str.lower().str.contains(needle, regex=False)
        ]
    if order == "Confidence thấp trước":
        filtered = filtered.sort_values(["confidence", "gold_name"])
    elif order == "Confidence cao trước":
        filtered = filtered.sort_values(["confidence", "gold_name"], ascending=[False, True])
    elif order == "IoU thấp trước":
        filtered = filtered.sort_values(["best_same_label_iou", "confidence"])
    else:
        filtered = filtered.sort_values(["gold_name", "confidence"])
    st.sidebar.metric("Cases", len(filtered))
    if filtered.empty:
        st.info("Không có case phù hợp bộ lọc.")
        st.stop()

    def option(row: pd.Series) -> str:
        verdict = "ĐÚNG" if bool(row["top1_correct"]) else "SAI"
        return (
            f"{verdict} · {row['scene_id']} · GT {row['gold_name']} · ours {row['ours_name']} · "
            f"conf {row['confidence']:.3f} · IoU {row['best_same_label_iou']:.2f}"
        )

    option_to_index = {
        option(row): int(row["index"]) for _, row in filtered.iterrows()
    }
    selected_option = st.selectbox("Case để nghe", list(option_to_index))
    selected_index = option_to_index[selected_option]
    prediction = predictions[selected_index]
    row = cases[cases["index"] == selected_index].iloc[0]
    scene = manifest[str(prediction["scene_id"])]
    gold_events = list(prediction.get("gold_events") or [])
    predicted_events = list(prediction.get("predicted_events") or [])
    top = top_prediction(predicted_events)
    audio_path = Path(str(scene["mixture_path"]))

    st.subheader("Audio + question + answer")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(f"**Audio:** `{prediction['scene_id']}`")
        if audio_path.is_file():
            st.audio(str(audio_path))
            st.caption(
                f"Thời lượng file thật: {float(scene['duration_seconds']):.3f} giây. "
                "Các clip dưới 1 giây có thể bị player làm tròn thành 0:00."
            )
        else:
            st.error(f"Không tìm thấy audio: {audio_path}")
        st.markdown("**Question:** What sound is present in this audio?")
    with right:
        verdict = "✅ ĐÚNG" if bool(row["top1_correct"]) else "❌ SAI"
        st.markdown(f"**Ours:** {row['ours_name']}")
        st.markdown(f"**Ground truth:** {row['gold_name']}")
        st.markdown(f"**Kết quả:** {verdict}")
        st.markdown(f"**Confidence:** {float(row['confidence']):.3f} ↑")
        st.markdown(
            "**Predicted evidence:** "
            + (format_span(top) if top is not None else "không có")
        )
        st.markdown(f"**Evidence IoU:** {float(row['best_same_label_iou']):.3f} ↑")

    st.subheader("GT và predicted evidence trên timeline")
    show_timeline(
        timeline_df(gold_events, predicted_events),
        duration_s=float(scene["duration_seconds"]),
    )
    gt_tab, predicted_tab, all_cases_tab = st.tabs(
        ["Ground truth", "Ours predicted inventory", "Bảng case"]
    )
    with gt_tab:
        st.dataframe(event_df(gold_events), hide_index=True, use_container_width=True)
    with predicted_tab:
        st.dataframe(
            event_df(predicted_events, include_confidence=True),
            hide_index=True,
            use_container_width=True,
        )
    with all_cases_tab:
        display = filtered[
            [
                "scene_id",
                "gold_name",
                "ours_name",
                "confidence",
                "top1_correct",
                "gold_detected",
                "best_same_label_iou",
                "predicted_events",
            ]
        ].copy()
        st.dataframe(display, hide_index=True, use_container_width=True)

    st.info(
        "Ở bản này evidence là đoạn thời gian detector cho rằng event đang hoạt động. "
        "AudioSep chỉ từng được dùng để tạo/lọc nguồn sạch; inference của checkpoint "
        "trên trang này không chạy AudioSep."
    )


if __name__ == "__main__":
    main()
