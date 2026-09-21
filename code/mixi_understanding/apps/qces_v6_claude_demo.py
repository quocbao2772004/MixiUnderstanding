#!/usr/bin/env python3
"""One-page Streamlit demo for Claude's QCES-v6 paper-scale artifacts.

The app is intentionally read-only: it loads the frozen receipts produced by the
QCES-v6 pipeline evaluation, shows question/planner/evidence diagnostics, and
plays already-rendered WAVs when they exist.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = PROJECT_ROOT / "outputs/qces_v6_results/pipeline_val/evaluation_report.json"
DEFAULT_EVENTWISE_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v6_eventwise_results/pipeline_val_eventwise/evaluation_report.json"
)
DEFAULT_EVENTWISE_SMOKE_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v6_eventwise_results/pipeline_val_eventwise_smoke64/evaluation_report.json"
)
DEFAULT_RELATION_THRESHOLD_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v6_rerank_results/pipeline_val_relation_threshold/evaluation_report.json"
)
DEFAULT_RELATION_THRESHOLD_SMOKE_REPORT = (
    PROJECT_ROOT
    / "outputs/qces_v6_rerank_results/pipeline_val_relation_threshold_smoke64/evaluation_report.json"
)
DEFAULT_DATASET = PROJECT_ROOT / "data/qces_v6_full_cropbank_v2"

MODE_ORDER = [
    "predicted__predicted_gate",
    "eventwise__predicted_gate",
    "predicted__no_gate",
    "eventwise__predicted_no_gate",
    "oracle_text__oracle_gate",
    "oracle_text__no_gate",
    "oracle_inventory__planned_gate",
    "question_prompt__no_gate",
    "energy_predicted__predicted_gate",
    "mixture_passthrough",
]

MODE_LABELS = {
    "predicted__predicted_gate": "Ours · predicted prompt + predicted gate",
    "eventwise__predicted_gate": "Ours event-wise · predicted events + gate",
    "predicted__no_gate": "Ours · predicted prompt, no gate",
    "eventwise__predicted_no_gate": "Ours event-wise · predicted events, no gate",
    "oracle_text__oracle_gate": "Ceiling · oracle text + oracle gate",
    "oracle_text__no_gate": "Ceiling · oracle text, no gate",
    "oracle_inventory__planned_gate": "Ceiling · oracle inventory + planned gate",
    "question_prompt__no_gate": "Baseline · raw question as prompt",
    "energy_predicted__predicted_gate": "Baseline · energy proposal + gate",
    "mixture_passthrough": "Mixture passthrough",
}

MODE_NOTES = {
    "predicted__predicted_gate": (
        "Deployable QCES-v6 path trong receipt: proposal head → symbolic planner "
        "→ text prompt → predicted temporal spans/gate → evidence."
    ),
    "eventwise__predicted_gate": (
        "Renderer mới: planner giữ nguyên như Claude, nhưng tách từng planned event "
        "bằng prompt đơn rồi gate từng event span trước khi cộng lại. Mục tiêu là giảm leakage từ prompt gộp."
    ),
    "predicted__no_gate": (
        "Cùng predicted prompt nhưng không cắt theo gate. Mode này thường nghe ít bị "
        "câm hơn, dùng để thấy bottleneck nằm ở temporal gate."
    ),
    "eventwise__predicted_no_gate": (
        "Event-wise không gate: dùng để kiểm tra prompt đơn theo từng event có sạch hơn prompt gộp không."
    ),
    "oracle_text__oracle_gate": (
        "Diagnostic ceiling: dùng text/spans từ annotation. Đây là mốc để nghe xem "
        "AudioSep + span đúng có thể gần target đến đâu."
    ),
    "oracle_text__no_gate": (
        "Diagnostic ceiling về semantic prompt: text đúng nhưng không có temporal gate."
    ),
    "oracle_inventory__planned_gate": (
        "Diagnostic ceiling: parser/planner được cấp inventory thật, không phải "
        "proposal dự đoán. Nếu mode này tốt còn ours tệ thì lỗi ở proposal."
    ),
    "question_prompt__no_gate": (
        "Baseline không học: đưa nguyên câu hỏi vào separator làm text prompt."
    ),
    "energy_predicted__predicted_gate": (
        "Baseline proposal bằng năng lượng stem, không dùng learned CLAP/event head."
    ),
    "mixture_passthrough": "Không tách gì; evidence = mixture.",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args, _ = parser.parse_known_args(argv)
    return args


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    value = float(value)
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def fmt_seconds(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "?"
    return f"{float(value):.2f}s"


def humanize(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value).replace("_", " ").strip()
    return " ".join(text.split()) or "—"


def direction_label(name: str) -> str:
    if name.endswith("_↑"):
        return f"{name[:-2]} ↑"
    if name.endswith("_↓"):
        return f"{name[:-2]} ↓"
    return name


def relation_requirement(relation: Any, question: Any) -> str:
    rel = str(relation or "").casefold()
    q = str(question or "").casefold()
    if not rel:
        if "before" in q:
            rel = "before"
        elif "after" in q or "subsequent" in q or "next" in q:
            rel = "after"
        elif "first" in q:
            rel = "first"
    if rel == "after":
        return "Yêu cầu: tìm event mốc được hỏi, rồi tìm event bắt đầu ngay sau mốc đó."
    if rel == "before":
        return "Yêu cầu: tìm event mốc được hỏi, rồi tìm event bắt đầu ngay trước mốc đó."
    if rel == "first":
        return "Yêu cầu: so sánh onset của các event ứng viên để chọn event xuất hiện đầu tiên."
    return "Yêu cầu: giữ phần âm thanh tối thiểu đủ để kiểm chứng câu trả lời."


@st.cache_data(show_spinner=False)
def load_json(path_text: str) -> dict[str, Any]:
    return json.loads(Path(path_text).expanduser().resolve().read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_jsonl_by_id(path_text: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path_text).expanduser().resolve().open("r", encoding="utf-8") as handle:
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
    sr, audio = wavfile.read(Path(path_text).expanduser().resolve())
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        denom = float(max(abs(info.min), abs(info.max)))
        audio = audio.astype(np.float32) / denom
    else:
        audio = audio.astype(np.float32, copy=False)
    return int(sr), np.nan_to_num(np.ascontiguousarray(audio, dtype=np.float32))


def wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    payload = BytesIO()
    wavfile.write(payload, sample_rate, np.asarray(audio, dtype=np.float32))
    return payload.getvalue()


def report_manifest_path(report: Mapping[str, Any]) -> Path:
    value = report.get("manifest")
    if isinstance(value, str) and value:
        return Path(value).expanduser().resolve()
    split = str(report.get("split", "val"))
    return DEFAULT_DATASET / f"qces_{split}.jsonl"


def resolve_dataset_path(manifest_path: Path, path_value: Any) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def output_question_dir(report_path: Path, item: Mapping[str, Any]) -> Path:
    return (
        report_path.parent
        / str(item.get("mode"))
        / str(item.get("scene_id"))
        / f"q{item.get('question_index')}_{item.get('question_type')}"
    )


def has_rendered_wavs(report_path: Path, item: Mapping[str, Any]) -> bool:
    qdir = output_question_dir(report_path, item)
    return (qdir / "predicted_evidence.wav").is_file() and (
        qdir / "predicted_residual.wav"
    ).is_file()


def group_report_items(report: Mapping[str, Any], report_path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in report.get("items", []):
        if not isinstance(item, dict):
            continue
        sample_id = item.get("id")
        if not isinstance(sample_id, str):
            continue
        copied = dict(item)
        copied["_rendered"] = has_rendered_wavs(report_path, copied)
        grouped[sample_id].append(copied)
    for sample_id, items in grouped.items():
        items.sort(key=lambda item: MODE_ORDER.index(item["mode"]) if item.get("mode") in MODE_ORDER else 99)
    return grouped


def role_for_event(event_id: str, row: Mapping[str, Any]) -> str:
    roles: list[str] = []
    if event_id in set(row.get("anchor_event_ids") or []):
        roles.append("anchor")
    if event_id in set(row.get("answer_event_ids") or []):
        roles.append("answer")
    if event_id in set(row.get("evidence_event_ids") or []):
        roles.append("evidence")
    if event_id in set(row.get("query_event_ids") or []):
        roles.append("query")
    return " + ".join(roles) if roles else "distractor"


def event_table(row: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in row.get("events", []) or []:
        if not isinstance(event, Mapping):
            continue
        event_id = str(event.get("event_id", ""))
        rows.append(
            {
                "event": event_id,
                "sound": f"{humanize(event.get('label'))} #{event.get('occurrence_index', 1)}",
                "time": (
                    f"{fmt_seconds(event.get('onset_seconds'))}"
                    f"–{fmt_seconds(event.get('offset_seconds'))}"
                ),
                "role": role_for_event(event_id, row),
                "source": event.get("source_id", ""),
            }
        )
    rows.sort(key=lambda item: item["time"])
    return pd.DataFrame(rows)


def event_map(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(event.get("event_id")): event
        for event in row.get("events", []) or []
        if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
    }


def build_target_stems(manifest_path: Path, row: Mapping[str, Any]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    mixture_path = resolve_dataset_path(manifest_path, row.get("mixture_path"))
    if mixture_path is None or not mixture_path.is_file():
        raise FileNotFoundError(f"Missing mixture WAV: {mixture_path}")
    sr, mixture = read_audio(str(mixture_path))
    evidence = np.zeros_like(mixture, dtype=np.float32)
    events = event_map(row)
    for event_id in row.get("evidence_event_ids", []) or []:
        event = events.get(str(event_id))
        if event is None:
            continue
        stem_path = resolve_dataset_path(manifest_path, event.get("stem_path"))
        if stem_path is None or not stem_path.is_file():
            continue
        stem_sr, stem = read_audio(str(stem_path))
        if stem_sr != sr:
            continue
        n = min(evidence.size, stem.size)
        evidence[:n] += stem[:n]
    return sr, mixture, evidence, mixture - evidence


def span_text(spans: Any) -> str:
    if not isinstance(spans, list) or not spans:
        return "—"
    parts: list[str] = []
    for span in spans:
        if isinstance(span, (list, tuple)) and len(span) == 2:
            parts.append(f"{fmt_seconds(span[0])}–{fmt_seconds(span[1])}")
    return ", ".join(parts) if parts else "—"


def item_metric_rows(items: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        metrics = item.get("metrics", {})
        descriptives = item.get("descriptives", {})
        rows.append(
            {
                "mode": MODE_LABELS.get(str(item.get("mode")), str(item.get("mode"))),
                "answer": humanize(item.get("planned_answer_label")),
                "answer ok ↑": item.get("planner_answer_correct"),
                "no-ev ok ↑": item.get("planner_no_evidence_correct"),
                "IoU ↑": fmt_float(item.get("planner_span_iou_↑")),
                "SD-SDRi ↑": fmt_float(metrics.get("evidence_sd_sdri_db_↑")),
                "SI-SDRi ↑": fmt_float(metrics.get("evidence_si_sdri_db_↑")),
                "retained ↓": fmt_float(descriptives.get("evidence_retained_ratio")),
                "pred spans": span_text(item.get("predicted_spans")),
                "WAV": "yes" if item.get("_rendered") else "no",
            }
        )
    return pd.DataFrame(rows)


def summary_rows(report: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summaries = report.get("summaries_by_mode", {})
    if not isinstance(summaries, Mapping):
        return pd.DataFrame()
    for mode in MODE_ORDER:
        summary = summaries.get(mode)
        if not isinstance(summary, Mapping):
            continue
        rows.append(
            {
                "mode": MODE_LABELS.get(mode, mode),
                "records": summary.get("record_count"),
                "answer ↑": fmt_float(summary.get("planner_answer_accuracy_↑")),
                "no-evid ↑": fmt_float(summary.get("planner_no_evidence_accuracy_↑")),
                "IoU ↑": fmt_float(summary.get("planner_span_iou_mean_↑")),
                "SD-SDRi mean ↑": fmt_float(summary.get("evidence_sd_sdri_answerable_mean_db_↑")),
                "SD-SDR median ↑": fmt_float(summary.get("evidence_sd_sdr_answerable_median_db_↑")),
                "positive rate ↑": fmt_float(summary.get("evidence_sd_sdri_answerable_positive_rate_↑")),
                "abstain ↓": fmt_float(summary.get("planner_abstention_rate_on_answerable_↓")),
            }
        )
    return pd.DataFrame(rows)


def sample_label(sample_id: str, items: Sequence[Mapping[str, Any]]) -> str:
    item = items[0]
    primary = primary_item(items)
    if primary.get("no_evidence"):
        status = "noev-ok" if primary.get("planner_no_evidence_correct") else "noev-wrong"
    else:
        status = "answer-ok" if primary.get("planner_answer_correct") else "answer-wrong"
    wav = "audio" if any(x.get("_rendered") for x in items) else "metric-only"
    answer = humanize(item.get("answer"))
    rel = item.get("relation", "?")
    qidx = item.get("question_index", "?")
    scene = item.get("scene_id", "?")
    question = str(item.get("question", "")).strip()
    if len(question) > 74:
        question = question[:71] + "..."
    return (
        f"q{qidx} {rel} · {question} · gold={answer} · "
        f"{status} · {wav} · {scene}"
    )


def primary_item(items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for mode in ("eventwise__predicted_gate", "predicted__predicted_gate"):
        found = next((item for item in items if item.get("mode") == mode), None)
        if found is not None:
            return found
    return items[0]


OUTCOME_OPTIONS = {
    "all": "all",
    "answer_correct": "answer correct",
    "answer_wrong": "answer wrong",
    "noev_correct": "no-evidence correct",
    "noev_wrong": "no-evidence wrong",
    "answerable": "answerable only",
    "no_evidence": "no-evidence only",
}


def filter_sample_ids(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    row_by_id: Mapping[str, Mapping[str, Any]],
    relation: str,
    outcome: str,
    wav_only: bool,
    query: str,
) -> list[str]:
    query_folded = query.casefold().strip()
    ids: list[str] = []
    for sample_id, items in grouped.items():
        first = items[0]
        if wav_only and not any(item.get("_rendered") for item in items):
            continue
        if relation != "all" and str(first.get("relation")) != relation:
            continue
        pred = primary_item(items)
        is_no_evidence = bool(pred.get("no_evidence"))
        answer_correct = pred.get("planner_answer_correct")
        noev_correct = bool(pred.get("planner_no_evidence_correct"))
        if outcome == "answerable" and is_no_evidence:
            continue
        if outcome == "no_evidence" and not is_no_evidence:
            continue
        if outcome == "answer_correct" and (
            is_no_evidence or answer_correct is not True
        ):
            continue
        if outcome == "answer_wrong" and (
            is_no_evidence or answer_correct is not False
        ):
            continue
        if outcome == "noev_correct" and (not is_no_evidence or not noev_correct):
            continue
        if outcome == "noev_wrong" and (not is_no_evidence or noev_correct):
            continue
        row = row_by_id.get(sample_id, {})
        haystack = " ".join(
            [
                sample_id,
                str(first.get("scene_id", "")),
                str(first.get("question", "")),
                str(first.get("answer", "")),
                " ".join(str(event.get("label", "")) for event in row.get("events", []) if isinstance(event, Mapping)),
            ]
        ).casefold()
        if query_folded and query_folded not in haystack:
            continue
        ids.append(sample_id)
    return sorted(ids)


def render_event_audio_grid(manifest_path: Path, row: Mapping[str, Any], max_events: int) -> None:
    events = [event for event in row.get("events", []) or [] if isinstance(event, Mapping)]
    if not events:
        st.info("Scene này không có event inventory.")
        return
    st.caption("Mỗi player bên dưới là crop đúng onset–offset đã hiển thị trên UI, lấy từ aligned event stem 10s.")
    for idx, event in enumerate(events[:max_events]):
        event_id = str(event.get("event_id", ""))
        label = humanize(event.get("label"))
        onset = float(event.get("onset_seconds", 0.0) or 0.0)
        offset = float(event.get("offset_seconds", onset) or onset)
        stem_path = resolve_dataset_path(manifest_path, event.get("stem_path"))
        with st.expander(
            f"{event_id} · {label} #{event.get('occurrence_index', 1)} · "
            f"{onset:.2f}–{offset:.2f}s · {role_for_event(event_id, row)}",
            expanded=idx < 2,
        ):
            if stem_path is None or not stem_path.is_file():
                st.warning(f"Không thấy stem: {stem_path}")
                continue
            sr, stem = read_audio(str(stem_path))
            start = max(0, min(int(round(onset * sr)), stem.size))
            stop = max(start, min(int(round(offset * sr)), stem.size))
            crop = stem[start:stop]
            st.audio(wav_bytes(sr, crop), format="audio/wav")
            st.caption(f"Source: `{event.get('source_id', '')}` · `{stem_path.name}`")


def main() -> None:
    args = parse_args()

    st.set_page_config(
        page_title="QCES-v6 Claude receipt demo",
        page_icon="🎧",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .qces-demo-card {
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 0.75rem;
            padding: 1rem 1.1rem;
            background: rgba(128, 128, 128, 0.08);
            margin: 0.6rem 0 1.0rem 0;
        }
        .qces-question {
            font-size: 1.35rem;
            line-height: 1.45;
            font-weight: 700;
            margin: 0.2rem 0 0.55rem 0;
        }
        .qces-demo-small {
            color: rgba(128, 128, 128, 0.95);
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("QCES-v6 · demo từ kết quả Claude")
    st.caption(
        "Một trang để nghe và debug: Audio + Question → planner chọn prompt/span → "
        "evidence/residual. Đây là receipt demo, không chạy inference mới."
    )

    report_options: list[tuple[str, Path]] = []
    requested_report = args.report.expanduser().resolve()
    if requested_report.is_file():
        report_options.append(("Claude original val", requested_report))
    if DEFAULT_EVENTWISE_REPORT.is_file():
        report_options.append(("Event-wise val", DEFAULT_EVENTWISE_REPORT.resolve()))
    if DEFAULT_RELATION_THRESHOLD_REPORT.is_file():
        report_options.append(("Relation-threshold rerank val", DEFAULT_RELATION_THRESHOLD_REPORT.resolve()))
    if DEFAULT_EVENTWISE_SMOKE_REPORT.is_file():
        report_options.append(("Event-wise smoke64", DEFAULT_EVENTWISE_SMOKE_REPORT.resolve()))
    if DEFAULT_RELATION_THRESHOLD_SMOKE_REPORT.is_file():
        report_options.append(("Relation-threshold smoke64", DEFAULT_RELATION_THRESHOLD_SMOKE_REPORT.resolve()))
    if not report_options:
        report_options.append(("Requested report", requested_report))
    deduped: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, path in report_options:
        if path in seen:
            continue
        seen.add(path)
        deduped.append((label, path))
    report_options = deduped
    selected_report_index = st.radio(
        "Chọn receipt",
        range(len(report_options)),
        format_func=lambda index: report_options[index][0],
        horizontal=True,
    )
    report_name, report_path = report_options[selected_report_index]

    if not report_path.is_file():
        st.error(f"Không thấy report: `{report_path}`")
        st.stop()

    report = load_json(str(report_path))
    manifest_path = report_manifest_path(report)
    if not manifest_path.is_file():
        st.error(f"Không thấy manifest: `{manifest_path}`")
        st.stop()
    row_by_id = load_jsonl_by_id(str(manifest_path))
    grouped = group_report_items(report, report_path)

    info_cols = st.columns(4)
    info_cols[0].metric("report records", str(report.get("record_count", "—")))
    info_cols[1].metric("manifest rows", str(len(row_by_id)))
    info_cols[2].metric("AudioSep frozen", str(report.get("audiosep_frozen", "—")))
    info_cols[3].metric("proposal threshold", str(report.get("proposal_threshold", "—")))
    st.caption(f"Receipt đang xem: `{report_name}`")
    st.caption(f"Report: `{report_path}`")
    st.caption(f"Manifest: `{manifest_path}`")

    with st.expander("Bảng tổng hợp split đang xem", expanded=True):
        st.dataframe(summary_rows(report), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Chọn case để nghe")
    controls = st.columns([0.7, 0.7, 0.7, 1.2])
    with controls[0]:
        relation = st.selectbox("Relation", ["all", "after", "before", "first"])
    with controls[1]:
        outcome = st.selectbox(
            "Outcome filter",
            list(OUTCOME_OPTIONS),
            format_func=lambda value: OUTCOME_OPTIONS[value],
        )
    with controls[2]:
        wav_only = st.checkbox("Chỉ case có WAV", value=True)
    with controls[3]:
        query = st.text_input("Search label/question/scene", value="")

    candidate_ids = filter_sample_ids(
        grouped,
        row_by_id=row_by_id,
        relation=relation,
        outcome=outcome,
        wav_only=wav_only,
        query=query,
    )
    if not candidate_ids:
        st.warning("Không có case phù hợp filter.")
        st.stop()

    selected_id = st.selectbox(
        f"Case ({len(candidate_ids)} phù hợp)",
        candidate_ids,
        format_func=lambda sample_id: sample_label(sample_id, grouped[sample_id]),
    )
    row = row_by_id.get(selected_id)
    if row is None:
        st.error(f"Manifest thiếu sample id: `{selected_id}`")
        st.stop()
    items = grouped[selected_id]
    first = items[0]
    ours = primary_item(items)

    gold_answer = humanize(first.get("answer"))
    pred_answer = humanize(ours.get("planned_answer_label"))
    if ours.get("no_evidence"):
        pred_status = (
            "NO-EVIDENCE ĐÚNG"
            if ours.get("planner_no_evidence_correct")
            else "NO-EVIDENCE SAI"
        )
    else:
        pred_status = "ANSWER ĐÚNG" if ours.get("planner_answer_correct") else "ANSWER SAI"
    st.markdown(
        f"""
        <div class="qces-demo-card">
          <div class="qces-demo-small">DEMO CASE: <code>{selected_id}</code> · scene <code>{first.get('scene_id')}</code></div>
          <div class="qces-question">Câu hỏi: {first.get('question')}</div>
          <div>Yêu cầu: {relation_requirement(first.get("relation"), first.get("question"))}</div>
          <div style="margin-top:0.55rem;">
            Gold answer: <code>{gold_answer}</code>
            &nbsp; | &nbsp;
            Ours predicted answer: <code>{pred_answer}</code>
            &nbsp; | &nbsp;
            Kết quả planner: <code>{pred_status}</code>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Question → requirement → prediction")
    qcols = st.columns([1.4, 1.0, 1.0])
    with qcols[0]:
        st.markdown(f"Question: `{first.get('question')}`")
        st.write(relation_requirement(first.get("relation"), first.get("question")))
        st.markdown(f"Gold answer: `{humanize(first.get('answer'))}`")
        st.markdown(f"Answer options: `{', '.join(humanize(x) for x in row.get('answer_options', []))}`")
    with qcols[1]:
        st.markdown("Ours planner")
        st.markdown(f"Predicted answer: `{humanize(ours.get('planned_answer_label'))}`")
        st.markdown(f"Prompt: `{ours.get('prompt') or 'no_evidence'}`")
        st.markdown(f"Predicted spans: `{span_text(ours.get('predicted_spans'))}`")
        st.markdown(f"Reason: `{ours.get('planner_reason')}`")
    with qcols[2]:
        st.markdown("Gold evidence")
        st.markdown(f"Evidence labels: `{', '.join(humanize(x) for x in first.get('gold_evidence_labels', [])) or 'no_evidence'}`")
        st.markdown(f"Anchor ids: `{', '.join(row.get('anchor_event_ids') or []) or '—'}`")
        st.markdown(f"Answer ids: `{', '.join(row.get('answer_event_ids') or []) or '—'}`")
        st.markdown(f"Evidence ids: `{', '.join(row.get('evidence_event_ids') or []) or '—'}`")

    st.subheader("Scene inventory")
    st.dataframe(event_table(row), use_container_width=True, hide_index=True)

    st.subheader("Audio gốc và target stems")
    try:
        sr, mixture, target_evidence, target_residual = build_target_stems(manifest_path, row)
        acols = st.columns(3)
        with acols[0]:
            st.markdown("Mixture X")
            st.audio(wav_bytes(sr, mixture), format="audio/wav")
        with acols[1]:
            st.markdown("Target evidence E*")
            st.audio(wav_bytes(sr, target_evidence), format="audio/wav")
        with acols[2]:
            st.markdown("Target residual R* = X - E*")
            st.audio(wav_bytes(sr, target_residual), format="audio/wav")
    except Exception as exc:  # pragma: no cover - Streamlit-facing diagnostics.
        st.error(f"Không build được target stems: {exc}")

    st.subheader("So sánh modes")
    st.dataframe(item_metric_rows(items), use_container_width=True, hide_index=True)

    tabs = st.tabs([MODE_LABELS.get(str(item.get("mode")), str(item.get("mode"))) for item in items])
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
                st.info("Mode này có metric trong receipt nhưng chưa render WAV trong thư mục output.")
            with st.expander("Raw item JSON", expanded=False):
                compact = {
                    "id": item.get("id"),
                    "mode": item.get("mode"),
                    "prompt": item.get("prompt"),
                    "planned_answer_label": item.get("planned_answer_label"),
                    "planned_labels": item.get("planned_labels"),
                    "planned_no_evidence": item.get("planned_no_evidence"),
                    "planner_reason": item.get("planner_reason"),
                    "predicted_spans": item.get("predicted_spans"),
                    "metrics": {direction_label(k): v for k, v in (item.get("metrics") or {}).items()},
                    "descriptives": item.get("descriptives"),
                }
                st.json(compact, expanded=False)

    st.subheader("Nghe từng loại âm trong scene")
    max_events = st.slider("Số event crop hiển thị", min_value=1, max_value=16, value=min(8, len(row.get("events", []) or [])))
    render_event_audio_grid(manifest_path, row, max_events=max_events)

    st.info(
        "Cách check nhanh: nếu `Ceiling · oracle text + oracle gate` gần Target evidence nhưng "
        "`Ours · predicted prompt + predicted gate` tệ, lỗi nằm ở proposal/span/gate chứ không phải dataset."
    )


if __name__ == "__main__":
    main()
