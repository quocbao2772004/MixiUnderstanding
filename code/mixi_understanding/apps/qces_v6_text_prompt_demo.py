#!/usr/bin/env python3
"""One-page Streamlit demo for the QCES-v6 text-prompt AudioSep diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import streamlit as st
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v5_pilot300_diverse_seed2028/text_prompt_ablation_val/evaluation_report.json"
)
DEFAULT_STRUCTURED_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v5_pilot300_diverse_seed2028/structured_planner_val/evaluation_report.json"
)
DEFAULT_TEMPORAL_GROUNDER_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v5_pilot300_diverse_seed2028/temporal_grounder_audiosep_train/evaluation_report.json"
)
DEFAULT_TEMPORAL_POSTPROCESS_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v5_pilot300_diverse_seed2028/temporal_grounder_postprocess_val/evaluation_report.json"
)

MODE_LABELS = {
    "candidate_best_text__oracle_gate": "Text prompt + oracle temporal gate",
    "candidate_best_text__no_gate": "Text prompt only / no temporal gate",
    "candidate_best_text__qces_predicted_gate": "Text prompt + QCES predicted gate",
    "structured_planner_text__planned_event_gate": "Structured planner text + planned event gate",
    "structured_planner_text__no_gate": "Structured planner text only / no temporal gate",
    "temporal_grounder__soft_gate": "Temporal grounder · soft gate",
    "temporal_grounder__hard_gate_train_threshold": "Temporal grounder · hard gate",
    "temporal_grounder__train_calibrated_gate": "Temporal grounder · train-calibrated gate",
}
MODE_NOTES = {
    "candidate_best_text__oracle_gate": (
        "Ceiling diagnostic: prompt text đúng manifold + thời gian oracle từ annotation."
    ),
    "candidate_best_text__no_gate": (
        "Ablation: AudioSep nhận text prompt nhưng không biết đoạn thời gian cần giữ."
    ),
    "candidate_best_text__qces_predicted_gate": (
        "Diagnostic hiện tại: dùng gate học từ checkpoint cũ; nếu tệ hơn oracle gate thì bottleneck nằm ở temporal grounding."
    ),
    "structured_planner_text__planned_event_gate": (
        "Planner không dùng target waveform/answer để chọn prompt; nó dùng structured question fields + event inventory để chọn events và spans."
    ),
    "structured_planner_text__no_gate": (
        "Cùng prompt của structured planner nhưng không dùng temporal gate; dùng để đo riêng tác dụng của temporal grounding."
    ),
    "temporal_grounder__soft_gate": (
        "Gate học từ raw AudioSep evidence features; dùng sigmoid probability nhân trực tiếp vào waveform."
    ),
    "temporal_grounder__hard_gate_train_threshold": (
        "Gate học rồi hard-threshold bằng ngưỡng chọn trên train frame-F1."
    ),
    "temporal_grounder__train_calibrated_gate": (
        "Postprocess candidate chọn trên train waveform metric rồi áp vào val."
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--structured-report", type=Path, default=DEFAULT_STRUCTURED_REPORT)
    parser.add_argument("--temporal-grounder-report", type=Path, default=DEFAULT_TEMPORAL_GROUNDER_REPORT)
    parser.add_argument("--temporal-postprocess-report", type=Path, default=DEFAULT_TEMPORAL_POSTPROCESS_REPORT)
    args, _ = parser.parse_known_args(argv)
    return args


def humanize(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    return " ".join(value.replace("_", " ").strip().split()) or "unknown"


def fmt_seconds(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "?"
    return f"{float(value):.2f}s"


def fmt_number(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def relation_requirement(relation: object, question: object = "") -> str:
    normalized = str(relation or "").strip().casefold()
    folded_question = str(question or "").casefold()
    if not normalized:
        if "before" in folded_question:
            normalized = "before"
        elif "after" in folded_question or "next" in folded_question:
            normalized = "after"
        elif "first" in folded_question:
            normalized = "first"
    if normalized == "after":
        return "Yêu cầu: tìm âm mốc trong câu hỏi, rồi giữ bằng chứng gồm âm mốc và âm bắt đầu ngay sau nó."
    if normalized == "before":
        return "Yêu cầu: tìm âm mốc trong câu hỏi, rồi giữ bằng chứng gồm âm ngay trước nó và âm mốc."
    if normalized == "first":
        return "Yêu cầu: so sánh onset của các âm ứng viên; evidence phải giữ đủ các âm cần so sánh."
    return "Yêu cầu: giữ đoạn âm thanh tối thiểu đủ để kiểm chứng câu trả lời."


@st.cache_data(show_spinner=False)
def load_report(report_path_text: str) -> dict[str, Any]:
    report_path = Path(report_path_text).expanduser().resolve()
    return json.loads(report_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_manifest(manifest_path_text: str) -> dict[str, dict[str, Any]]:
    manifest_path = Path(manifest_path_text).expanduser().resolve()
    rows: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            if isinstance(sample_id, str):
                rows[sample_id] = row
    return rows


@st.cache_data(show_spinner=False)
def read_audio(path_text: str) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(Path(path_text).expanduser().resolve())
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        denom = max(abs(info.min), abs(info.max))
        audio = audio.astype(np.float32) / float(denom)
    else:
        audio = audio.astype(np.float32, copy=False)
    audio = np.nan_to_num(audio, copy=False)
    return int(sample_rate), np.ascontiguousarray(audio, dtype=np.float32)


def wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    payload = BytesIO()
    wavfile.write(payload, sample_rate, np.asarray(audio, dtype=np.float32))
    return payload.getvalue()


def resolve_dataset_path(manifest_path: Path, relative_or_absolute: object) -> Path | None:
    if not isinstance(relative_or_absolute, str) or not relative_or_absolute:
        return None
    path = Path(relative_or_absolute)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def event_by_id(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    events = row.get("events", [])
    if not isinstance(events, list):
        return {}
    return {
        str(event.get("event_id")): event
        for event in events
        if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
    }


def role_for_event(event_id: str, row: Mapping[str, Any]) -> str:
    anchors = set(row.get("anchor_event_ids", []) or [])
    answers = set(row.get("answer_event_ids", []) or [])
    evidence = set(row.get("evidence_event_ids", []) or [])
    roles: list[str] = []
    if event_id in anchors:
        roles.append("anchor")
    if event_id in answers:
        roles.append("answer")
    if event_id in evidence:
        roles.append("evidence")
    return " + ".join(roles) if roles else "distractor"


def event_table(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = row.get("events", [])
    if not isinstance(events, list):
        return []
    table: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_id = str(event.get("event_id", ""))
        label = humanize(event.get("label"))
        occurrence = event.get("occurrence_index", 1)
        table.append(
            {
                "event_id": event_id,
                "sound": f"{label} #{occurrence}",
                "time": f"{fmt_seconds(event.get('onset_seconds'))}–{fmt_seconds(event.get('offset_seconds'))}",
                "role": role_for_event(event_id, row),
                "source": event.get("source_id", ""),
            }
        )
    return sorted(table, key=lambda item: item["time"])


def build_target_audio(
    *,
    manifest_path: Path,
    row: Mapping[str, Any],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    mixture_path = resolve_dataset_path(manifest_path, row.get("mixture_path"))
    if mixture_path is None or not mixture_path.is_file():
        raise FileNotFoundError(f"missing mixture: {mixture_path}")
    sample_rate, mixture = read_audio(str(mixture_path))
    evidence = np.zeros_like(mixture, dtype=np.float32)
    events = event_by_id(row)
    for event_id in row.get("evidence_event_ids", []) or []:
        event = events.get(str(event_id))
        if event is None:
            continue
        stem_path = resolve_dataset_path(manifest_path, event.get("stem_path"))
        if stem_path is None or not stem_path.is_file():
            continue
        stem_rate, stem = read_audio(str(stem_path))
        if stem_rate != sample_rate:
            continue
        n = min(evidence.size, stem.size)
        evidence[:n] += stem[:n]
    residual = mixture - evidence
    return sample_rate, mixture, evidence, residual


def output_question_dir(report_path: Path, item: Mapping[str, Any]) -> Path:
    split = item.get("split_evaluated")
    if isinstance(split, str) and split:
        return (
            report_path.parent
            / split
            / str(item["mode"])
            / str(item["scene_id"])
            / f"q{item['question_index']}_{item['question_type']}"
        )
    return (
        report_path.parent
        / str(item["mode"])
        / str(item["scene_id"])
        / f"q{item['question_index']}_{item['question_type']}"
    )


def has_rendered_prediction(report_path: Path, item: Mapping[str, Any]) -> bool:
    question_dir = output_question_dir(report_path, item)
    return (question_dir / "predicted_evidence.wav").is_file() and (
        question_dir / "predicted_residual.wav"
    ).is_file()


def metric_rows(items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        metrics = item.get("metrics", {})
        descriptives = item.get("descriptives", {})
        rows.append(
            {
                "mode": MODE_LABELS.get(str(item.get("mode")), str(item.get("mode"))),
                "SD-SDRi ↑": fmt_number(metrics.get("evidence_sd_sdri_db_↑")),
                "SI-SDRi ↑": fmt_number(metrics.get("evidence_si_sdri_db_↑")),
                "Evidence L1 ↓": fmt_number(metrics.get("evidence_l1_↓"), 4),
                "Retained energy ↓/diag": fmt_number(
                    descriptives.get("evidence_retained_ratio"), 3
                ),
                "Pred audio": "yes" if item.get("_rendered") else "no",
            }
        )
    return rows


def summary_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    summaries = report.get("summaries_by_mode", {})
    if not summaries and isinstance(report.get("summaries_by_split"), Mapping):
        summaries = report["summaries_by_split"].get("val", {})
    if not isinstance(summaries, Mapping):
        return rows
    for mode, summary in summaries.items():
        if not isinstance(summary, Mapping):
            continue
        rows.append(
            {
                "mode": MODE_LABELS.get(str(mode), str(mode)),
                "answerable": str(summary.get("answerable_count", "—")),
                "SD-SDRi mean ↑": fmt_number(
                    summary.get("evidence_sd_sdri_answerable_mean_db_↑")
                ),
                "SI-SDRi mean ↑": fmt_number(
                    summary.get("evidence_si_sdri_answerable_mean_db_↑")
                ),
                "Evidence L1 ↓": fmt_number(summary.get("evidence_l1_mean_↓"), 4),
                "No-evidence retained ↓": fmt_number(
                    summary.get("no_evidence_retained_ratio_mean_↓"), 4
                ),
            }
        )
    return rows


def report_manifest_path(report: Mapping[str, Any]) -> Path:
    value = report.get("manifest") or report.get("val_manifest")
    if not isinstance(value, str):
        raise ValueError("report does not declare manifest/val_manifest")
    return Path(value).expanduser().resolve()


def report_items(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = report.get("items", [])
    if isinstance(items, list):
        return [dict(item) for item in items if isinstance(item, dict)]
    if isinstance(items, Mapping):
        val_items = items.get("val", [])
        if isinstance(val_items, list):
            return [dict(item) for item in val_items if isinstance(item, dict)]
    return []


def main() -> None:
    args = parse_args()
    st.set_page_config(
        page_title="QCES-v6 text prompt demo",
        page_icon="🎧",
        layout="wide",
    )
    st.title("QCES-v6 · Text prompt AudioSep diagnostic")
    report_options: list[tuple[str, Path]] = []
    default_report = args.report.expanduser().resolve()
    structured_report = args.structured_report.expanduser().resolve()
    temporal_grounder_report = args.temporal_grounder_report.expanduser().resolve()
    temporal_postprocess_report = args.temporal_postprocess_report.expanduser().resolve()
    if default_report.is_file():
        report_options.append(("Oracle text diagnostic", default_report))
    if structured_report.is_file():
        report_options.append(("Structured planner baseline", structured_report))
    if temporal_grounder_report.is_file():
        report_options.append(("Temporal grounder", temporal_grounder_report))
    if temporal_postprocess_report.is_file():
        report_options.append(("Temporal grounder postprocess", temporal_postprocess_report))
    if not report_options:
        st.error(
            "Không thấy report nào: "
            f"{default_report} hoặc {structured_report}"
        )
        st.stop()

    selected_report_index = st.radio(
        "Chọn experiment",
        range(len(report_options)),
        format_func=lambda index: report_options[index][0],
        horizontal=True,
    )
    report_name, report_path = report_options[selected_report_index]
    report = load_report(str(report_path))
    manifest_path = report_manifest_path(report)
    manifest = load_manifest(str(manifest_path))

    st.caption(
        f"Đang xem `{report_name}`. Demo một trang: question → canonical text prompt "
        "→ AudioSep evidence/residual. Mode có `oracle gate` là diagnostic ceiling, "
        "chưa phải deployable model."
    )
    st.dataframe(summary_rows(report), use_container_width=True, hide_index=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in report_items(report):
        item["_rendered"] = has_rendered_prediction(report_path, item)
        grouped[str(item["id"])].append(item)
    rendered_ids = {
        sample_id for sample_id, items in grouped.items() if any(item["_rendered"] for item in items)
    }

    st.divider()
    left, right = st.columns([1.0, 1.4])
    with left:
        only_rendered = st.checkbox(
            "Chỉ hiện case có predicted WAV đã render",
            value=True,
            help="Tắt option này để xem text/metric cho toàn bộ 48 val records.",
        )
        candidate_ids = sorted(rendered_ids if only_rendered else grouped)
        if not candidate_ids:
            st.warning("Không có case nào phù hợp filter.")
            st.stop()

        def option_label(sample_id: str) -> str:
            item = grouped[sample_id][0]
            marker = "audio" if sample_id in rendered_ids else "metric-only"
            labels = ", ".join(humanize(label) for label in item.get("target_labels", []))
            if not labels:
                labels = "no evidence"
            return (
                f"{sample_id} · q{item.get('question_index')} "
                f"{item.get('relation')} · {labels} · {marker}"
            )

        selected_id = st.selectbox(
            "Chọn case",
            candidate_ids,
            index=0,
            format_func=option_label,
        )

    items = sorted(grouped[selected_id], key=lambda item: str(item.get("mode")))
    row = manifest.get(selected_id)
    if row is None:
        st.error(f"Manifest không có ID: {selected_id}")
        st.stop()

    with right:
        first = items[0]
        st.subheader("Câu hỏi và yêu cầu")
        st.write(f"Question: `{first.get('question')}`")
        st.write(f"Gold answer: `{humanize(first.get('answer'))}`")
        st.write(relation_requirement(first.get("relation"), first.get("question")))
        st.write(f"Selected AudioSep prompt: `{first.get('prompt') or 'no evidence'}`")
        target_labels = ", ".join(humanize(label) for label in first.get("target_labels", []))
        st.write(f"Target evidence labels: `{target_labels or 'no_evidence'}`")

    st.subheader("Scene events")
    st.dataframe(event_table(row), use_container_width=True, hide_index=True)

    sample_rate, mixture, target_evidence, target_residual = build_target_audio(
        manifest_path=manifest_path, row=row
    )
    st.subheader("Audio gốc và target")
    audio_cols = st.columns(3)
    with audio_cols[0]:
        st.markdown("Mixture X")
        st.audio(wav_bytes(sample_rate, mixture), format="audio/wav")
    with audio_cols[1]:
        st.markdown("Target evidence E*")
        st.audio(wav_bytes(sample_rate, target_evidence), format="audio/wav")
    with audio_cols[2]:
        st.markdown("Target residual R*")
        st.audio(wav_bytes(sample_rate, target_residual), format="audio/wav")

    st.subheader("Predicted evidence/residual theo từng mode")
    st.dataframe(metric_rows(items), use_container_width=True, hide_index=True)
    tabs = st.tabs([MODE_LABELS.get(str(item["mode"]), str(item["mode"])) for item in items])
    for tab, item in zip(tabs, items):
        with tab:
            st.write(MODE_NOTES.get(str(item.get("mode")), ""))
            qdir = output_question_dir(report_path, item)
            pred_e = qdir / "predicted_evidence.wav"
            pred_r = qdir / "predicted_residual.wav"
            if pred_e.is_file() and pred_r.is_file():
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("Predicted evidence Ê")
                    st.audio(pred_e.read_bytes(), format="audio/wav")
                with cols[1]:
                    st.markdown("Predicted residual R̂ = X - Ê")
                    st.audio(pred_r.read_bytes(), format="audio/wav")
            else:
                st.warning("Case này có metric nhưng chưa render predicted WAV.")
            st.json(
                {
                    "prompt": item.get("prompt"),
                    "metrics": item.get("metrics"),
                    "descriptives": item.get("descriptives"),
                },
                expanded=False,
            )

    with st.expander("Nghe từng event/crop đúng timestamp trong scene"):
        events = row.get("events", [])
        if not isinstance(events, list) or not events:
            st.info("Không có event inventory.")
        else:
            event_options = [
                event
                for event in events
                if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
            ]
            selected_event = st.selectbox(
                "Chọn event",
                event_options,
                format_func=lambda event: (
                    f"{event.get('event_id')} · {humanize(event.get('label'))} "
                    f"#{event.get('occurrence_index', 1)} · "
                    f"{fmt_seconds(event.get('onset_seconds'))}–{fmt_seconds(event.get('offset_seconds'))} · "
                    f"{role_for_event(str(event.get('event_id')), row)}"
                ),
            )
            stem_path = resolve_dataset_path(manifest_path, selected_event.get("stem_path"))
            if stem_path is None or not stem_path.is_file():
                st.warning(f"Không thấy stem: {stem_path}")
            else:
                event_rate, stem = read_audio(str(stem_path))
                onset = float(selected_event.get("onset_seconds", 0.0))
                offset = float(selected_event.get("offset_seconds", onset))
                start = max(0, min(int(round(onset * event_rate)), stem.size))
                stop = max(start, min(int(round(offset * event_rate)), stem.size))
                crop = stem[start:stop]
                st.write(
                    f"Crop đang nghe: `{humanize(selected_event.get('label'))}` "
                    f"từ `{onset:.2f}s` đến `{offset:.2f}s` trong mixture."
                )
                st.audio(wav_bytes(event_rate, crop), format="audio/wav")
                if st.checkbox("Nghe full aligned stem 10s của event này", value=False):
                    st.audio(wav_bytes(event_rate, stem), format="audio/wav")

    st.caption(
        "Cách check nhanh: so `Target evidence E*` với `Text prompt + oracle temporal gate`. "
        "Nếu oracle gate giống target nhưng QCES predicted gate tệ, lỗi chính là temporal gate."
    )


if __name__ == "__main__":
    main()
