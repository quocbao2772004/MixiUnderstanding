#!/usr/bin/env python3
"""Streamlit viewer for the QCES 200-class detector validation outputs.

This app is intentionally offline: it reads an existing detector run
(`training_report.json` + `val_predictions.jsonl`) and the matching detector
manifest. It does not run model inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover - Streamlit can still run without Altair.
    alt = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HEAD_ONLY_RUN_DIR = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e8_headonly"
)
FINETUNE_RUN_DIR = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e4_unfreeze1_highthr_resume"
)
DEFAULT_RUN_DIR = FINETUNE_RUN_DIR
KNOWN_RUNS = {
    "Fine-tune 1 block cuối, 4 epochs (latest)": FINETUNE_RUN_DIR,
    "Head-only 8 epochs baseline": HEAD_ONLY_RUN_DIR,
}
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/qces_multisource_detector_trainset_v1/detector_manifest_val.jsonl"
)
DEFAULT_DATASET_SUMMARY = (
    PROJECT_ROOT
    / "outputs/qces_multisource_detector_trainset_v1/multisource_detector_trainset_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-summary", type=Path, default=DEFAULT_DATASET_SUMMARY)
    parser.add_argument("--iou", type=float, default=0.30)
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def label_text(label: str | None) -> str:
    if not label:
        return ""
    return label.replace("_and_", " / ").replace("_", " ")


def event_name(event: dict[str, Any]) -> str:
    return str(event.get("display_name") or label_text(str(event.get("label", ""))))


def interval_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0 = float(a.get("onset_seconds", 0.0))
    a1 = float(a.get("offset_seconds", a0))
    b0 = float(b.get("onset_seconds", 0.0))
    b1 = float(b.get("offset_seconds", b0))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def match_events(
    gold_events: list[dict[str, Any]],
    pred_events: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy same-label matching by IoU/confidence for one scene."""
    candidates: list[tuple[float, float, int, int]] = []
    for gi, gold in enumerate(gold_events):
        for pi, pred in enumerate(pred_events):
            if gold.get("label") != pred.get("label"):
                continue
            iou = interval_iou(gold, pred)
            if iou >= iou_threshold:
                candidates.append((iou, float(pred.get("confidence", 1.0)), gi, pi))

    candidates.sort(reverse=True)
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, confidence, gi, pi in candidates:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        gold = gold_events[gi]
        pred = pred_events[pi]
        matches.append(
            {
                "label": gold.get("label"),
                "event": event_name(gold),
                "gt_span": format_span(gold),
                "pred_span": format_span(pred),
                "iou": iou,
                "confidence": confidence,
            }
        )

    false_pos = [pred for i, pred in enumerate(pred_events) if i not in used_pred]
    false_neg = [gold for i, gold in enumerate(gold_events) if i not in used_gold]
    return matches, false_pos, false_neg


def format_span(event: dict[str, Any]) -> str:
    onset = float(event.get("onset_seconds", 0.0))
    offset = float(event.get("offset_seconds", onset))
    return f"{onset:.2f}–{offset:.2f}s"


def event_df(events: list[dict[str, Any]], *, include_confidence: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        row: dict[str, Any] = {
            "#": idx,
            "event": event_name(event),
            "label": event.get("label", ""),
            "span": format_span(event),
            "onset_s": round(float(event.get("onset_seconds", 0.0)), 3),
            "offset_s": round(float(event.get("offset_seconds", 0.0)), 3),
        }
        if include_confidence:
            row["confidence ↑"] = round(float(event.get("confidence", 1.0)), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def scene_metrics(row: dict[str, Any], iou_threshold: float) -> dict[str, Any]:
    matches, fp, fn = match_events(
        list(row.get("gold_events", [])),
        list(row.get("predicted_events", [])),
        iou_threshold=iou_threshold,
    )
    tp = len(matches)
    precision = tp / max(tp + len(fp), 1)
    recall = tp / max(tp + len(fn), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "false_pos": fp,
        "false_neg": fn,
    }


def build_case_index(
    predictions: list[dict[str, Any]],
    manifest_by_scene: dict[str, dict[str, Any]],
    iou_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        scene_id = str(pred.get("scene_id", ""))
        manifest = manifest_by_scene.get(scene_id, {})
        metrics = scene_metrics(pred, iou_threshold)
        gold_labels = sorted({str(e.get("label")) for e in pred.get("gold_events", [])})
        pred_labels = sorted({str(e.get("label")) for e in pred.get("predicted_events", [])})
        rows.append(
            {
                "idx": idx,
                "scene_id": scene_id,
                "route": manifest.get("source_route", pred.get("split", "")),
                "duration_s": round(float(manifest.get("duration_seconds", 0.0)), 2),
                "gt_events": len(pred.get("gold_events", [])),
                "pred_events": len(pred.get("predicted_events", [])),
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "event_f1": metrics["f1"],
                "gt_labels": ", ".join(label_text(x) for x in gold_labels),
                "pred_labels": ", ".join(label_text(x) for x in pred_labels),
                "audio_exists": resolve_path(manifest.get("mixture_path", "")).exists()
                if manifest.get("mixture_path")
                else False,
            }
        )
    return pd.DataFrame(rows)


def latest_metrics(report: dict[str, Any]) -> dict[str, float]:
    selected = (report.get("val_detector") or {}).get("selected")
    if isinstance(selected, dict):
        scene_label = selected.get("scene_label", {})
        event_iou = selected.get("event_iou", {})
        frame = selected.get("frame", {})
        return {
            "Scene-label F1 ↑": float(scene_label.get("f1_↑", 0.0)),
            "Scene-label precision ↑": float(scene_label.get("precision_↑", 0.0)),
            "Scene-label recall ↑": float(scene_label.get("recall_↑", 0.0)),
            "Event F1 @ IoU0.30 ↑": float(event_iou.get("f1_↑", 0.0)),
            "Frame F1 ↑": float(frame.get("f1_↑", 0.0)),
            "Avg FP labels ↓": float(scene_label.get("avg_false_positive_labels_↓", 0.0)),
            "Mean onset error s ↓": float(event_iou.get("mean_onset_abs_error_s_↓", 0.0)),
            "Mean offset error s ↓": float(event_iou.get("mean_offset_abs_error_s_↓", 0.0)),
        }
    if isinstance(report.get("best_metrics"), dict):
        best = report["best_metrics"]
        return {
            "Scene-label F1 ↑": float(best.get("scene_label_f1_↑", 0.0)),
            "Scene-label precision ↑": float(best.get("scene_label_precision_↑", 0.0)),
            "Scene-label recall ↑": float(best.get("scene_label_recall_↑", 0.0)),
            "Event F1 @ IoU0.30 ↑": float(best.get("event_f1_↑", 0.0)),
            "Frame F1 ↑": float(best.get("frame_f1_↑", 0.0)),
            "Avg FP labels ↓": float(best.get("avg_false_positive_labels_↓", 0.0)),
        }
    history = report.get("history") or []
    row = history[-1] if history else {}
    return {
        "Scene-label F1 ↑": float(row.get("scene_label_f1_↑", 0.0)),
        "Scene-label precision ↑": float(row.get("scene_label_precision_↑", 0.0)),
        "Scene-label recall ↑": float(row.get("scene_label_recall_↑", 0.0)),
        "Event F1 @ IoU0.30 ↑": float(row.get("event_f1_↑", 0.0)),
        "Frame F1 ↑": float(row.get("frame_f1_↑", 0.0)),
        "Avg FP labels ↓": float(row.get("avg_false_positive_labels_↓", 0.0)),
    }


def metric_table(report: dict[str, Any]) -> pd.DataFrame:
    metrics = latest_metrics(report)
    rows = [{"metric": k, "value": round(v, 4)} for k, v in metrics.items()]
    rows.append({"metric": "Selected threshold", "value": report.get("best_threshold", "")})
    rows.append({"metric": "Train scenes", "value": report.get("train_scenes", report.get("train_rows", ""))})
    rows.append({"metric": "Val scenes", "value": report.get("val_scenes", report.get("val_rows", ""))})
    rows.append({"metric": "Ontology labels", "value": len(report.get("labels", [])) or report.get("num_labels", "")})
    return pd.DataFrame(rows)


def timeline_df(
    gold_events: list[dict[str, Any]], pred_events: list[dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in gold_events:
        rows.append(
            {
                "type": "GT",
                "event": event_name(event),
                "label": label_text(str(event.get("label", ""))),
                "start": float(event.get("onset_seconds", 0.0)),
                "end": float(event.get("offset_seconds", 0.0)),
                "confidence": 1.0,
            }
        )
    for event in pred_events:
        rows.append(
            {
                "type": "Predicted",
                "event": event_name(event),
                "label": label_text(str(event.get("label", ""))),
                "start": float(event.get("onset_seconds", 0.0)),
                "end": float(event.get("offset_seconds", 0.0)),
                "confidence": float(event.get("confidence", 1.0)),
            }
        )
    return pd.DataFrame(rows)


def show_timeline(df: pd.DataFrame, duration_s: float) -> None:
    if df.empty:
        st.info("Không có event để vẽ timeline.")
        return
    if alt is None:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("start:Q", title="time (s)", scale=alt.Scale(domain=[0, max(duration_s, 0.1)])),
            x2="end:Q",
            y=alt.Y("event:N", title=None, sort="-x"),
            color=alt.Color("type:N", title="track"),
            tooltip=["type", "event", "label", "start", "end", "confidence"],
        )
        .properties(height=max(180, min(620, 28 * df["event"].nunique() + 90)))
    )
    st.altair_chart(chart, use_container_width=True)


def filter_cases(index_df: pd.DataFrame) -> pd.DataFrame:
    routes = sorted(str(x) for x in index_df["route"].dropna().unique())
    selected_routes = st.sidebar.multiselect("Source route", routes, default=routes)
    case_type = st.sidebar.selectbox(
        "Case type",
        [
            "All",
            "Clean-ish: no FP/FN",
            "Has false positives",
            "Has false negatives",
            "Heavy FP: ≥3",
            "Low event F1: <0.25",
            "High event F1: ≥0.75",
        ],
    )
    label_query = st.sidebar.text_input("Search scene/label", "")
    sort_by = st.sidebar.selectbox(
        "Sort",
        [
            "worst event F1",
            "best event F1",
            "most false positives",
            "most false negatives",
            "most predicted events",
            "most GT events",
        ],
    )

    out = index_df[index_df["route"].astype(str).isin(selected_routes)].copy()
    if case_type == "Clean-ish: no FP/FN":
        out = out[(out["fp"] == 0) & (out["fn"] == 0)]
    elif case_type == "Has false positives":
        out = out[out["fp"] > 0]
    elif case_type == "Has false negatives":
        out = out[out["fn"] > 0]
    elif case_type == "Heavy FP: ≥3":
        out = out[out["fp"] >= 3]
    elif case_type == "Low event F1: <0.25":
        out = out[out["event_f1"] < 0.25]
    elif case_type == "High event F1: ≥0.75":
        out = out[out["event_f1"] >= 0.75]

    if label_query.strip():
        q = label_query.strip().lower()
        out = out[
            out["scene_id"].str.lower().str.contains(q, regex=False)
            | out["gt_labels"].str.lower().str.contains(q, regex=False)
            | out["pred_labels"].str.lower().str.contains(q, regex=False)
        ]

    if sort_by == "worst event F1":
        out = out.sort_values(["event_f1", "fp"], ascending=[True, False])
    elif sort_by == "best event F1":
        out = out.sort_values(["event_f1", "tp"], ascending=[False, False])
    elif sort_by == "most false positives":
        out = out.sort_values(["fp", "event_f1"], ascending=[False, True])
    elif sort_by == "most false negatives":
        out = out.sort_values(["fn", "event_f1"], ascending=[False, True])
    elif sort_by == "most predicted events":
        out = out.sort_values(["pred_events", "fp"], ascending=[False, False])
    elif sort_by == "most GT events":
        out = out.sort_values(["gt_events", "fn"], ascending=[False, False])
    return out


def format_case_option(row: pd.Series) -> str:
    return (
        f"{row['scene_id']} | F1={row['event_f1']:.2f} "
        f"TP/FP/FN={int(row['tp'])}/{int(row['fp'])}/{int(row['fn'])} | {row['route']}"
    )


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="QCES 200-class detector demo", layout="wide")
    st.title("QCES detector-first demo — 200 classes")
    st.caption(
        "Offline viewer cho detector 200-class: Mixture → pretrained SED head → event inventory "
        "(label + timestamp). Đây là bản head-only hiện tại, không phải fine-tune đang chạy."
    )

    run_choices = {name: path for name, path in KNOWN_RUNS.items() if path.exists()}
    requested_run_dir = args.run_dir.resolve()
    if requested_run_dir not in {p.resolve() for p in run_choices.values()}:
        run_choices[f"Custom: {requested_run_dir.name}"] = requested_run_dir
    default_index = 0
    for i, (_, path) in enumerate(run_choices.items()):
        if path.resolve() == requested_run_dir:
            default_index = i
            break
    st.sidebar.header("Detector run")
    selected_run_name = st.sidebar.selectbox(
        "Checkpoint/result để xem",
        list(run_choices.keys()),
        index=default_index,
    )
    run_dir = run_choices[selected_run_name].resolve()
    report_path = run_dir / "training_report.json"
    pred_path = run_dir / "val_predictions.jsonl"
    manifest_path = args.manifest.resolve()
    summary_path = args.dataset_summary.resolve()

    missing = [p for p in [report_path, pred_path, manifest_path] if not p.exists()]
    if missing:
        st.error("Thiếu file bắt buộc:\n" + "\n".join(str(p) for p in missing))
        st.stop()

    report = read_json(str(report_path))
    predictions = read_jsonl(str(pred_path))
    manifest_rows = read_jsonl(str(manifest_path))
    manifest_by_scene = {str(r.get("scene_id")): r for r in manifest_rows}
    dataset_summary = read_json(str(summary_path)) if summary_path.exists() else {}
    index_df = build_case_index(predictions, manifest_by_scene, args.iou)

    with st.expander("Benchmark + dataset summary", expanded=True):
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Detector metrics")
            st.caption(f"Run đang xem: {selected_run_name}")
            st.dataframe(metric_table(report), use_container_width=True, hide_index=True)
            st.caption("↑ càng cao càng tốt; ↓ càng thấp càng tốt.")
        with right:
            st.subheader("Dataset used by this detector")
            rows = []
            for split in ["train", "val", "test"]:
                item = dataset_summary.get(split, {})
                if item:
                    rows.append(
                        {
                            "split": split,
                            "scenes": item.get("rows"),
                            "events": item.get("events"),
                            "labels with positive": item.get("labels_with_positive"),
                            "missing labels": ", ".join(item.get("missing_labels", [])),
                            "routes": json.dumps(item.get("route_counts", {}), ensure_ascii=False),
                        }
                    )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Không tìm thấy dataset summary.")

    st.sidebar.header("Case browser")
    filtered = filter_cases(index_df)
    st.sidebar.metric("Cases after filter", len(filtered))
    if filtered.empty:
        st.warning("Không có case nào sau filter.")
        st.stop()

    st.subheader("Validation case")
    options = {format_case_option(row): int(row["idx"]) for _, row in filtered.iterrows()}
    chosen_label = st.selectbox("Chọn case để nghe/check", list(options.keys()))
    chosen_idx = options[chosen_label]
    pred_row = predictions[chosen_idx]
    scene_id = str(pred_row.get("scene_id"))
    manifest = manifest_by_scene.get(scene_id, {})
    metrics = scene_metrics(pred_row, args.iou)

    audio_path = resolve_path(manifest.get("mixture_path", ""))
    duration_s = float(manifest.get("duration_seconds", 0.0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GT events", len(pred_row.get("gold_events", [])))
    c2.metric("Pred events", len(pred_row.get("predicted_events", [])))
    c3.metric("TP ↑", metrics["tp"])
    c4.metric("FP ↓", metrics["fp"])
    c5.metric("FN ↓", metrics["fn"])

    st.markdown(
        f"""
**Scene:** `{scene_id}`  
**Source route:** `{manifest.get("source_route", "")}`  
**Duration:** `{duration_s:.2f}s`  
**GT labels:** {", ".join(label_text(x) for x in sorted(set(manifest.get("labels", []))))}
"""
    )

    if audio_path.exists():
        st.audio(str(audio_path))
    else:
        st.error(f"Không tìm thấy audio: {audio_path}")

    st.subheader("Timeline")
    st.caption("GT là annotation thật; Predicted là event inventory do detector dự đoán.")
    show_timeline(
        timeline_df(
            list(pred_row.get("gold_events", [])),
            list(pred_row.get("predicted_events", [])),
        ),
        duration_s=duration_s,
    )

    tab_gt, tab_pred, tab_match, tab_index = st.tabs(
        ["GT events", "Predicted events", "TP / FP / FN", "Case index"]
    )
    with tab_gt:
        st.dataframe(
            event_df(list(pred_row.get("gold_events", []))),
            use_container_width=True,
            hide_index=True,
        )
    with tab_pred:
        st.dataframe(
            event_df(list(pred_row.get("predicted_events", [])), include_confidence=True),
            use_container_width=True,
            hide_index=True,
        )
    with tab_match:
        left, mid, right = st.columns(3)
        with left:
            st.markdown("**TP matched events**")
            st.dataframe(
                pd.DataFrame(metrics["matches"]),
                use_container_width=True,
                hide_index=True,
            )
        with mid:
            st.markdown("**FP predicted only**")
            st.dataframe(
                event_df(metrics["false_pos"], include_confidence=True),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.markdown("**FN missed GT**")
            st.dataframe(
                event_df(metrics["false_neg"]),
                use_container_width=True,
                hide_index=True,
            )
    with tab_index:
        st.dataframe(
            filtered[
                [
                    "scene_id",
                    "route",
                    "duration_s",
                    "event_f1",
                    "tp",
                    "fp",
                    "fn",
                    "gt_events",
                    "pred_events",
                    "gt_labels",
                    "pred_labels",
                    "audio_exists",
                ]
            ].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "Cách đọc nhanh: nếu FP cao thì detector đang hallucinate class không có; "
        "nếu FN cao thì bỏ sót event thật; nếu đúng label nhưng timeline lệch thì IoU thấp."
    )


if __name__ == "__main__":
    main()
