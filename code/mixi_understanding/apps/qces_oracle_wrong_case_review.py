#!/usr/bin/env python3
"""Minimal Streamlit reviewer for oracle-evidence AudioQA failure cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "data/qces_v6_full_cropbank_v2"

MANIFEST = (
    PROJECT_ROOT
    / "outputs/qces_v6_oracle_target_evidence_from_pointer/fast384_answer_clean_stem"
    / "oracle_answer_clean_stem__oracle_answer_span_prompt/manifest.jsonl"
)
ORIGINAL_MANIFEST = (
    PROJECT_ROOT / "outputs/qces_v6_external_audioqa/fast384/qces_val_stratified384.jsonl"
)
PRED_ROOT_OURS = (
    PROJECT_ROOT
    / "outputs/qces_v6_predicted_pointer_audio/fast384_context_span_text"
    / "predicted_context_span_text/predictions"
)
PRED_ROOT_MASK = (
    PROJECT_ROOT
    / "outputs/qces_v6_oracle_target_evidence_from_pointer/fast384_context_span_text_clean"
    / "oracle_target_evidence_span__oracle_answer_span_prompt/predictions"
)
PRED_ROOT_CLEAN_STEM = (
    PROJECT_ROOT
    / "outputs/qces_v6_oracle_target_evidence_from_pointer/fast384_clean_stem"
    / "oracle_clean_stem__oracle_answer_span_prompt/predictions"
)
PRED_ROOT_ANSWER_ONLY = (
    PROJECT_ROOT
    / "outputs/qces_v6_oracle_target_evidence_from_pointer/fast384_answer_clean_stem"
    / "oracle_answer_clean_stem__oracle_answer_span_prompt/predictions"
)

RESULT_PATHS = {
    "AF3 · predicted evidence ours": PROJECT_ROOT
    / "outputs/qces_v6_external_audioqa/fast384/af3_context_span_text"
    / "span_text_mixture_and_predicted_evidence/items.jsonl",
    "AF3 · C mixture-mask": PROJECT_ROOT
    / "outputs/qces_v6_external_audioqa/fast384_oracle_target_evidence_clean"
    / "af3_oracle_target_evidence_clean/items.jsonl",
    "AF3 · C anchor+answer clean stem": PROJECT_ROOT
    / "outputs/qces_v6_external_audioqa/fast384_oracle_clean_stem"
    / "af3_oracle_clean_stem/items.jsonl",
    "AF3 · D answer-only clean stem": PROJECT_ROOT
    / "outputs/qces_v6_external_audioqa/fast384_oracle_answer_clean_stem"
    / "af3_oracle_answer_clean_stem/items.jsonl",
    "Qwen2-Audio · D answer-only clean stem": PROJECT_ROOT
    / "outputs/qces_v6_external_audioqa/fast384_oracle_answer_clean_stem"
    / "qwen2_audio_oracle_answer_clean_stem/items.jsonl",
    "Phi-4mm · D answer-only clean stem": PROJECT_ROOT
    / "outputs/qces_v6_external_audioqa/fast384_oracle_answer_clean_stem"
    / "phi4mm_oracle_answer_clean_stem/items.jsonl",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--original-manifest", type=Path, default=ORIGINAL_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args, _ = parser.parse_known_args(argv)
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fingerprint(paths: Sequence[Path]) -> str:
    parts = []
    for path in paths:
        if path.is_file():
            stat = path.stat()
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        else:
            parts.append(f"{path}:missing")
    return "|".join(parts)


def label_text(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("_", " ")


def fmt_span(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "—"
    return f"{start:.2f}–{end:.2f}s"


def safe_child(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"unsafe path: {path}")
    return path


def qdir(pred_root: Path, record: Mapping[str, Any]) -> Path:
    return (
        pred_root
        / str(record["scene_id"])
        / f"q{record['question_index']}_{record['question_type']}"
    )


@st.cache_data(show_spinner=False)
def load_bundle(
    manifest: str,
    original_manifest: str,
    dataset_root: str,
    artifact_state: str,
) -> dict[str, Any]:
    _ = artifact_state
    records = {str(row["id"]): row for row in read_jsonl(Path(manifest))}
    original_records = {
        str(row["id"]): row for row in read_jsonl(Path(original_manifest))
    }
    for sample_id, record in records.items():
        original_record = original_records.get(sample_id)
        if original_record:
            record["original_question"] = original_record.get("question")
            record["original_question_type"] = original_record.get("question_type")
            record["original_relation"] = original_record.get("relation")
            record["original_answer_options"] = original_record.get("answer_options")
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in RESULT_PATHS.items():
        by_id = {}
        for row in read_jsonl(path):
            if row.get("condition") == "predicted_evidence":
                by_id[str(row["id"])] = row
        results[name] = by_id
    return {
        "records": records,
        "results": results,
        "dataset_root": dataset_root,
    }


def acc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(bool(row.get("correct")) for row in rows) / len(rows)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def summary_table(results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> pd.DataFrame:
    rows = []
    for name, table in results.items():
        vals = list(table.values())
        ans = [row for row in vals if not row.get("no_evidence")]
        noev = [row for row in vals if row.get("no_evidence")]
        rows.append(
            {
                "result": name,
                "N": len(vals),
                "All ↑": pct(acc(vals)),
                "Answerable ↑": pct(acc(ans)),
                "No-evidence ↑": pct(acc(noev)),
                "Wrong answerable": sum(
                    (not row.get("no_evidence")) and (not bool(row.get("correct")))
                    for row in vals
                ),
            }
        )
    return pd.DataFrame(rows)


def event_rows(record: Mapping[str, Any], ids_key: str) -> list[dict[str, Any]]:
    wanted = {str(event_id) for event_id in record.get(ids_key, [])}
    rows = []
    for event in record.get("events", []):
        if str(event.get("event_id")) not in wanted:
            continue
        rows.append(
            {
                "event_id": event.get("event_id"),
                "label": label_text(event.get("label")),
                "occ": event.get("occurrence_index"),
                "span": fmt_span(
                    float(event.get("onset_seconds")),
                    float(event.get("offset_seconds")),
                ),
                "source_id": event.get("source_id"),
                "source_path": event.get("source_path"),
            }
        )
    return rows


def all_event_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for event in sorted(
        record.get("events", []), key=lambda item: float(item.get("onset_seconds", 0.0))
    ):
        rows.append(
            {
                "event_id": event.get("event_id"),
                "label": label_text(event.get("label")),
                "occ": event.get("occurrence_index"),
                "span": fmt_span(
                    float(event.get("onset_seconds")),
                    float(event.get("offset_seconds")),
                ),
                "source_id": event.get("source_id"),
            }
        )
    return rows


def option_table(record: Mapping[str, Any], result: Mapping[str, Any] | None) -> pd.DataFrame:
    probs = result.get("option_probabilities") if result else None
    rows = []
    for index, option in enumerate(record.get("answer_options", [])):
        row = {
            "option": chr(ord("A") + index),
            "label": label_text(option),
            "gold": str(option) == str(record.get("answer")),
        }
        if isinstance(probs, list) and index < len(probs):
            row["prob"] = f"{float(probs[index]):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def result_table(
    sample_id: str,
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> pd.DataFrame:
    rows = []
    for name, table in results.items():
        row = table.get(sample_id)
        if not row:
            rows.append(
                {
                    "result": name,
                    "prediction": "not run",
                    "correct": "—",
                    "gold_probability": "—",
                }
            )
            continue
        rows.append(
            {
                "result": name,
                "prediction": label_text(row.get("predicted_answer")),
                "correct": bool(row.get("correct")),
                "gold_probability": f"{float(row.get('gold_option_probability', 0.0)):.3f}",
            }
        )
    return pd.DataFrame(rows)


def result_for(
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    backend: str,
    sample_id: str,
) -> Mapping[str, Any] | None:
    return results.get(backend, {}).get(sample_id)


def audio_player(title: str, path: Path, caption: str = "") -> None:
    st.caption(title)
    if path.is_file():
        st.audio(str(path))
        if caption:
            st.caption(caption)
    else:
        st.warning(f"missing: {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    st.set_page_config(page_title="QCES wrong-case review", layout="wide")
    st.title("QCES oracle-evidence wrong-case review")
    st.caption(
        "Trang debug nhanh: xem case sai khi audio đã là answer-only clean stem, "
        "so với C mixture-mask / C clean-stem / predicted evidence."
    )

    if st.sidebar.button("Reload / clear cache"):
        st.cache_data.clear()

    paths = [args.manifest, args.original_manifest, *RESULT_PATHS.values()]
    bundle = load_bundle(
        str(args.manifest),
        str(args.original_manifest),
        str(args.dataset_root),
        fingerprint(paths),
    )
    records: dict[str, dict[str, Any]] = bundle["records"]
    results: dict[str, dict[str, dict[str, Any]]] = bundle["results"]
    if not records:
        st.error("Không load được manifest.")
        st.stop()

    st.subheader("Metrics")
    st.dataframe(summary_table(results), use_container_width=True, hide_index=True)

    d_backends = [
        "AF3 · D answer-only clean stem",
        "Qwen2-Audio · D answer-only clean stem",
        "Phi-4mm · D answer-only clean stem",
    ]

    with st.sidebar:
        st.header("Filters")
        backend = st.selectbox("Backend để lọc", d_backends, index=0)
        outcome = st.selectbox(
            "Outcome",
            [
                "wrong answerable",
                "wrong any",
                "correct",
                "all run",
                "not run",
                "wrong by any D backend",
                "wrong by all available D backends",
            ],
            index=0,
        )
        relation = st.selectbox(
            "Relation",
            ["all"] + sorted({str(record.get("relation")) for record in records.values()}),
            index=0,
        )
        labels = ["all"] + sorted(
            {str(record.get("answer")) for record in records.values() if record.get("answer")}
        )
        label = st.selectbox("Gold label", labels, index=0)
        search = st.text_input("Search id/source/pred", "")

    filtered: list[dict[str, Any]] = []
    search_norm = search.strip().lower()
    for sample_id, record in records.items():
        row = result_for(results, backend, sample_id)
        available_d = [
            result_for(results, name, sample_id)
            for name in d_backends
            if result_for(results, name, sample_id) is not None
        ]
        if outcome == "wrong answerable":
            keep = bool(row) and (not record.get("no_evidence")) and (not row.get("correct"))
        elif outcome == "wrong any":
            keep = bool(row) and (not row.get("correct"))
        elif outcome == "correct":
            keep = bool(row) and bool(row.get("correct"))
        elif outcome == "all run":
            keep = bool(row)
        elif outcome == "not run":
            keep = row is None
        elif outcome == "wrong by any D backend":
            keep = bool(available_d) and any(not backend_row.get("correct") for backend_row in available_d)
        elif outcome == "wrong by all available D backends":
            keep = bool(available_d) and all(not backend_row.get("correct") for backend_row in available_d)
        else:
            keep = True
        if not keep:
            continue
        if relation != "all" and str(record.get("relation")) != relation:
            continue
        if label != "all" and str(record.get("answer")) != label:
            continue
        if search_norm:
            haystack = " ".join(
                [
                    sample_id,
                    str(record.get("question")),
                    str(record.get("answer")),
                    " ".join(str(x) for x in record.get("answer_options", [])),
                    " ".join(
                        str(row.get("predicted_answer"))
                        for table in results.values()
                        for row in [table.get(sample_id)]
                        if row
                    ),
                    " ".join(
                        str(event.get("source_id")) + " " + str(event.get("source_path"))
                        for event in record.get("events", [])
                    ),
                ]
            ).lower()
            if search_norm not in haystack:
                continue
        filtered.append(record)

    st.subheader("Case selection")
    st.write(f"Showing {len(filtered)} / {len(records)} cases.")
    if not filtered:
        st.stop()

    def case_label(record: Mapping[str, Any]) -> str:
        sample_id = str(record["id"])
        row = result_for(results, backend, sample_id)
        pred = label_text(row.get("predicted_answer")) if row else "not run"
        ok = "✓" if row and row.get("correct") else "✗" if row else "…"
        return (
            f"{sample_id} · {record.get('relation')} · gold={label_text(record.get('answer'))} "
            f"· {backend.split(' · ')[0]}={pred} {ok}"
        )

    selected = st.selectbox("Case", [case_label(record) for record in filtered])
    record = filtered[[case_label(item) for item in filtered].index(selected)]
    sample_id = str(record["id"])
    selected_result = result_for(results, backend, sample_id)

    left, right = st.columns([1.1, 1.0])
    with left:
        st.markdown("Original question")
        st.info(str(record.get("original_question") or "—"))
        st.markdown("Diagnostic prompt used for this oracle-evidence eval")
        st.info(str(record.get("question")))
        st.markdown("Gold / selected backend")
        st.dataframe(
            pd.DataFrame(
                [
                    {"item": "Gold", "value": label_text(record.get("answer"))},
                    {
                        "item": backend,
                        "value": (
                            f"{label_text(selected_result.get('predicted_answer'))} "
                            f"({'correct' if selected_result.get('correct') else 'wrong'})"
                            if selected_result
                            else "not run"
                        ),
                    },
                    {"item": "Relation", "value": record.get("relation")},
                    {"item": "No-evidence", "value": bool(record.get("no_evidence"))},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("Answer options")
        st.dataframe(
            option_table(record, selected_result),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Predictions")
    st.dataframe(result_table(sample_id, results), use_container_width=True, hide_index=True)

    st.subheader("Evidence/source metadata")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("Answer event ids / stems")
        st.dataframe(
            pd.DataFrame(event_rows(record, "answer_event_ids")),
            use_container_width=True,
            hide_index=True,
        )
    with e2:
        st.markdown("Target evidence ids: anchor + answer")
        st.dataframe(
            pd.DataFrame(event_rows(record, "evidence_event_ids")),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Audio")
    mix_path = safe_child(Path(bundle["dataset_root"]), str(record["mixture_path"]))
    audio_specs = [
        ("Mixture original", mix_path, "audio gốc"),
        (
            "Predicted evidence ours",
            qdir(PRED_ROOT_OURS, record) / "predicted_evidence.wav",
            "output deployable hiện tại",
        ),
        (
            "C mixture-mask oracle evidence",
            qdir(PRED_ROOT_MASK, record) / "predicted_evidence.wav",
            "mask mixture theo gold target evidence span",
        ),
        (
            "C clean-stem anchor+answer",
            qdir(PRED_ROOT_CLEAN_STEM, record) / "predicted_evidence.wav",
            "clean stem của anchor + answer",
        ),
        (
            "D answer-only clean stem",
            qdir(PRED_ROOT_ANSWER_ONLY, record) / "predicted_evidence.wav",
            "chỉ clean stem của answer",
        ),
    ]
    cols = st.columns(2)
    for index, (title, path, caption) in enumerate(audio_specs):
        with cols[index % 2]:
            audio_player(title, path, caption)

    with st.expander("All scene events"):
        st.dataframe(pd.DataFrame(all_event_rows(record)), use_container_width=True, hide_index=True)
    with st.expander("Raw record/result"):
        st.json(
            {
                "record": record,
                "results": {
                    name: table.get(sample_id)
                    for name, table in results.items()
                    if table.get(sample_id) is not None
                },
            }
        )


if __name__ == "__main__":
    main(sys.argv[1:])
