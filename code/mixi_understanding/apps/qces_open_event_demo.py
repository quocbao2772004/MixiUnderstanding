#!/usr/bin/env python3
"""Streamlit demo for open-ish event QA with QCES full-scene proposals.

This is intentionally a sidecar app.  It does not modify the current Claude
demo or the paper evaluation code.

The Streamlit process can run in the lightweight base environment.  Torch/Qwen
work is executed in the qces-sam environment via subprocess, matching the
dependency split used by the existing demos.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import streamlit as st
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
INFER_SCRIPT = CODE_ROOT / "mixi_understanding/scripts/infer_qces_open_event_qa.py"

DEFAULT_CACHE = PROJECT_ROOT / "outputs/qces_v6_caches/val_stem_cache.pt"
DEFAULT_PROPOSAL_HEAD = (
    PROJECT_ROOT / "outputs/qces_v6_proposal_head_iou_v2/train_val_v1/proposal_head.pt"
)
DEFAULT_QWEN_MODEL = (
    "/home/cuongpv/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/"
    "snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a"
)
DEFAULT_EMBEDDING_MODEL = (
    "/home/cuongpv/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)
DEFAULT_QCES_PYTHON = Path("/home/cuongpv/anaconda3/envs/qces-sam/bin/python")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--proposal-head", type=Path, default=DEFAULT_PROPOSAL_HEAD)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--qces-python", type=Path, default=DEFAULT_QCES_PYTHON)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-events", type=int, default=40)
    args, _ = parser.parse_known_args(argv)
    return args


def command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(CODE_ROOT) if not current else str(CODE_ROOT) + os.pathsep + current
    )
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    return environment


def extract_json_object(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object found in subprocess output:\n{stdout[-2000:]}")
    return json.loads(stdout[start : end + 1])


def run_json_command(command: Sequence[str], timeout: int) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(PROJECT_ROOT),
        env=command_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "subprocess failed\n\n"
            f"command: {' '.join(command)}\n\n"
            f"stdout:\n{completed.stdout[-4000:]}\n\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    return extract_json_object(completed.stdout)


@st.cache_data(show_spinner=False)
def load_cache_metadata(cache_path_text: str, qces_python_text: str) -> dict[str, Any]:
    code = """
import json
import sys
from pathlib import Path
import torch

cache = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
scene_ids = list(cache["scenes"].keys())
labels_by_scene = {scene_id: list(entry["labels"]) for scene_id, entry in cache["scenes"].items()}
payload = {
    "format": cache.get("format"),
    "manifest": cache.get("manifest"),
    "manifest_sha256": cache.get("manifest_sha256"),
    "num_scenes": len(scene_ids),
    "scene_ids": scene_ids,
    "labels_by_scene": labels_by_scene,
    "label_counts": [len(labels_by_scene[scene_id]) for scene_id in scene_ids],
}
print(json.dumps(payload, ensure_ascii=False))
"""
    return run_json_command([qces_python_text, "-c", code, cache_path_text], timeout=120)


@st.cache_data(show_spinner=False)
def load_manifest_rows(manifest_path_text: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(manifest_path_text).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return tuple(rows)


@st.cache_data(show_spinner=False)
def run_open_qa(
    *,
    qces_python_text: str,
    cache_path_text: str,
    proposal_head_text: str,
    qwen_model_text: str,
    embedding_model_text: str,
    scene_id: str,
    question: str,
    parser_name: str,
    threshold: float,
    max_events: int,
    max_list_events: int,
    device: str,
) -> dict[str, Any]:
    command = [
        qces_python_text,
        str(INFER_SCRIPT),
        "--scene-id",
        scene_id,
        "--question",
        question,
        "--cache",
        cache_path_text,
        "--proposal-head",
        proposal_head_text,
        "--threshold",
        f"{threshold:.4f}",
        "--max-events",
        str(max_events),
        "--max-list-events",
        str(max_list_events),
        "--device",
        device,
        "--model",
        qwen_model_text,
        "--embedding-model",
        embedding_model_text,
        "--parser",
        parser_name,
    ]
    if parser_name == "rule":
        command.append("--no-llm")
    return run_json_command(command, timeout=180 if parser_name == "rule" else 300)


def first_row_by_scene(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        scene_id = row.get("scene_id")
        if isinstance(scene_id, str) and scene_id not in result:
            result[scene_id] = row
    return result


def human_label(label: str) -> str:
    return label.replace("_", " ")


def resolve_dataset_path(manifest_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def event_to_row(event: Mapping[str, Any], idx: int) -> dict[str, Any]:
    return {
        "#": idx + 1,
        "label": human_label(str(event.get("label", ""))),
        "label_raw": str(event.get("label", "")),
        "time": f"{float(event.get('onset_seconds', 0.0)):.2f}–{float(event.get('offset_seconds', 0.0)):.2f}s",
        "onset": round(float(event.get("onset_seconds", 0.0)), 3),
        "offset": round(float(event.get("offset_seconds", 0.0)), 3),
        "source": str(event.get("source_dataset", "")),
    }


def proposal_to_row(event: Mapping[str, Any], idx: int) -> dict[str, Any]:
    return {
        "#": idx + 1,
        "label": human_label(str(event.get("label", ""))),
        "label_raw": str(event.get("label", "")),
        "time": f"{float(event.get('onset_seconds', 0.0)):.2f}–{float(event.get('offset_seconds', 0.0)):.2f}s",
        "onset": round(float(event.get("onset_seconds", 0.0)), 3),
        "offset": round(float(event.get("offset_seconds", 0.0)), 3),
        "confidence": round(float(event.get("confidence", 0.0)), 4),
    }


def evidence_to_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    answer_keys = {
        (
            str(event.get("label", "")),
            round(float(event.get("onset_seconds", 0.0)), 3),
            round(float(event.get("offset_seconds", 0.0)), 3),
        )
        for event in (payload.get("answer_events") or [])
    }
    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(payload.get("evidence") or []):
        row = proposal_to_row(event, idx)
        key = (
            str(event.get("label", "")),
            round(float(event.get("onset_seconds", 0.0)), 3),
            round(float(event.get("offset_seconds", 0.0)), 3),
        )
        row["role"] = "answer" if key in answer_keys else "context/boundary"
        rows.append(row)
    return rows


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    audio_array = np.asarray(audio)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if np.issubdtype(audio_array.dtype, np.integer):
        info = np.iinfo(audio_array.dtype)
        audio_array = audio_array.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio_array = audio_array.astype(np.float32)
    return int(sample_rate), audio_array


def wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    payload = io.BytesIO()
    safe = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(safe))) if safe.size else 0.0
    if peak > 1.0:
        safe = safe / peak
    wavfile.write(payload, sample_rate, safe)
    return payload.getvalue()


def make_span_masked_audio(
    audio: np.ndarray,
    sample_rate: int,
    evidence: Sequence[Mapping[str, Any]],
    pad_seconds: float,
) -> np.ndarray:
    mask = np.zeros(len(audio), dtype=np.float32)
    for event in evidence:
        onset = float(event.get("onset_seconds", 0.0))
        offset = float(event.get("offset_seconds", 0.0))
        start = max(0, int(round((onset - pad_seconds) * sample_rate)))
        end = min(len(audio), int(round((offset + pad_seconds) * sample_rate)))
        if end > start:
            mask[start:end] = 1.0
    return np.asarray(audio, dtype=np.float32) * mask


def default_question(labels: Sequence[str]) -> str:
    if labels:
        return f"What starts right after the first {labels[0]}?"
    return "What sounds are in this audio?"


def example_questions(labels: Sequence[str]) -> list[str]:
    first = labels[0] if labels else "Camera"
    second = labels[1] if len(labels) > 1 else first
    return [
        "What sounds are in this audio?",
        f"Is there a {human_label(first)} sound?",
        f"How many times does {human_label(first)} occur?",
        f"What starts right after the first {first}?",
        f"What happens right before the first {second}?",
        f"What sound is between the third {first} sound and {second}?",
        "Which sound lasts the longest?",
        f"Between {first} and {second}, which starts earlier?",
    ]


def render_event_audio_grid(
    manifest_path: Path,
    events: Sequence[Mapping[str, Any]],
    max_items: int,
) -> None:
    if max_items <= 0:
        return
    st.caption(f"Nghe riêng tối đa {max_items} GT event đầu tiên.")
    for event in events[:max_items]:
        label = human_label(str(event.get("label", "")))
        stem_path = resolve_dataset_path(manifest_path, event.get("stem_path"))
        cols = st.columns([1, 4])
        cols[0].write(
            f"{label}\n\n{float(event.get('onset_seconds', 0.0)):.2f}–{float(event.get('offset_seconds', 0.0)):.2f}s"
        )
        if stem_path is not None and stem_path.is_file():
            cols[1].audio(stem_path.read_bytes(), format="audio/wav")
        else:
            cols[1].warning("missing stem wav")


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="QCES Open Event QA", layout="wide")
    st.title("QCES open-event QA demo")
    st.caption(
        "Luồng demo: free-form question → rule/embedding/Qwen parser → QCES full-scene event inventory → symbolic answer + predicted evidence spans."
    )

    with st.spinner("Loading QCES cache metadata..."):
        metadata = load_cache_metadata(str(args.cache.resolve()), str(args.qces_python.resolve()))
    manifest_path = Path(metadata["manifest"]).expanduser().resolve()
    rows = load_manifest_rows(str(manifest_path))
    scene_rows = first_row_by_scene(rows)
    scene_ids = [scene_id for scene_id in metadata["scene_ids"] if scene_id in scene_rows]
    label_counts = [int(x) for x in metadata["label_counts"]]

    with st.sidebar:
        st.header("Config")
        st.caption("Threshold thấp: recall cao hơn nhưng noisy. Threshold cao: sạch hơn nhưng dễ miss event.")
        threshold = st.slider("Inventory threshold", 0.05, 0.80, 0.20, 0.01)
        max_events = st.slider("Max predicted events", 5, 80, args.max_events, 1)
        max_gt_audio = st.slider("GT event audio clips", 0, 20, 6, 1)
        evidence_pad = st.slider("Evidence audio padding (s)", 0.0, 0.50, 0.05, 0.01)
        parser_mode = st.radio(
            "Question parser",
            ["Rule parser (fast)", "Embedding parser (semantic intent)", "Qwen parser (free-form, slower)"],
            index=1,
        )
        st.divider()
        st.write("Supported question families:")
        st.markdown(
            "- list events\n"
            "- yes/no presence\n"
            "- count occurrences\n"
            "- first / last / longest\n"
            "- before / after anchor event\n"
            "- between two anchor events\n"
            "- compare which starts earlier"
        )

    split_cols = st.columns(4)
    split_cols[0].metric("Scenes", len(scene_ids))
    split_cols[1].metric("Median labels / scene", int(np.median(label_counts)) if label_counts else 0)
    split_cols[2].metric("Min–max labels / scene", f"{min(label_counts)}–{max(label_counts)}" if label_counts else "0")
    split_cols[3].metric("Manifest rows", len(rows))

    selected_scene = st.selectbox("Scene", scene_ids, index=0)
    labels = list(metadata["labels_by_scene"][selected_scene])
    row = scene_rows[selected_scene]
    events = list(row.get("events") or [])

    with st.spinner("Running QCES full-inventory proposal head..."):
        inventory_payload = run_open_qa(
            qces_python_text=str(args.qces_python.resolve()),
            cache_path_text=str(args.cache.resolve()),
            proposal_head_text=str(args.proposal_head.resolve()),
            qwen_model_text=args.qwen_model,
            embedding_model_text=args.embedding_model,
            scene_id=selected_scene,
            question="What sounds are in this audio?",
            parser_name="rule",
            threshold=threshold,
            max_events=max_events,
            max_list_events=max_events,
            device=args.device,
        )
    predicted_events = list(inventory_payload.get("inventory_preview") or [])

    top_cols = st.columns([1.15, 1])
    with top_cols[0]:
        st.subheader("Audio gốc")
        mixture_path = resolve_dataset_path(manifest_path, row.get("mixture_path"))
        if mixture_path is None or not mixture_path.is_file():
            st.error(f"Missing mixture wav: {mixture_path}")
            return
        sample_rate, mixture = read_audio(mixture_path)
        st.audio(wav_bytes(sample_rate, mixture), format="audio/wav")
        st.caption(f"Scene `{selected_scene}` · duration {float(row.get('duration_seconds', 10.0)):.1f}s")

    with top_cols[1]:
        st.subheader("Example questions")
        for question_text in example_questions(labels[:4]):
            st.code(question_text, language="text")

    st.subheader("Hỏi tự do trên nhiều event của scene")
    question = st.text_input(
        "Question",
        value=default_question(labels),
        help="Parser sẽ map câu hỏi về chương trình có cấu trúc rồi chạy trên full event inventory.",
    )
    run = st.button("Run QCES open-event QA", type="primary")

    gt_tab, pred_tab, qa_tab = st.tabs(
        ["GT scene events", "Predicted full inventory", "Answer + evidence"]
    )

    with gt_tab:
        st.write(
            "Đây là event inventory thật từ synthetic manifest. Dùng để kiểm tra audio gồm những sự kiện gì."
        )
        if events:
            st.dataframe(
                [event_to_row(event, idx) for idx, event in enumerate(events)],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Nghe từng GT event stem"):
                render_event_audio_grid(manifest_path, events, max_gt_audio)
        else:
            st.info("Scene này không có GT events trong manifest row.")

    with pred_tab:
        st.write(
            "Đây là inventory do model của mình dự đoán từ full label set của scene, không phải 4 lựa chọn."
        )
        st.dataframe(
            [proposal_to_row(event, idx) for idx, event in enumerate(predicted_events)],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Available labels in cache: {len(labels)} · predicted proposals after threshold: {len(predicted_events)}"
        )

    with qa_tab:
        if not run:
            st.info("Nhập câu hỏi rồi bấm Run.")
            return
        parser_name = (
            "rule"
            if parser_mode.startswith("Rule")
            else "embedding"
            if parser_mode.startswith("Embedding")
            else "qwen"
        )
        with st.spinner("Parsing question and executing over predicted inventory..."):
            payload = run_open_qa(
                qces_python_text=str(args.qces_python.resolve()),
                cache_path_text=str(args.cache.resolve()),
                proposal_head_text=str(args.proposal_head.resolve()),
                qwen_model_text=args.qwen_model,
                embedding_model_text=args.embedding_model,
                scene_id=selected_scene,
                question=question,
                parser_name=parser_name,
                threshold=threshold,
                max_events=max_events,
                max_list_events=max_events,
                device=args.device,
            )

        program = payload.get("program") or {}
        answer_cols = st.columns(4)
        answer_cols[0].metric("Ours answer", str(payload.get("answer", "")))
        answer_cols[1].metric("No evidence?", "yes" if payload.get("no_evidence") else "no")
        answer_cols[2].metric("Operation", str(program.get("operation", "")))
        answer_cols[3].metric("Evidence spans", len(payload.get("evidence") or []))

        st.write("Parsed program")
        st.json(program)
        if payload.get("llm_raw"):
            with st.expander("Parser raw output"):
                st.code(str(payload.get("llm_raw")), language="json")
                st.write(payload.get("llm_json"))

        st.write("Predicted evidence")
        evidence = list(payload.get("evidence") or [])
        st.dataframe(
            evidence_to_rows(payload),
            use_container_width=True,
            hide_index=True,
        )
        if payload.get("answer_events"):
            st.caption(
                "`answer` = event mà executor chọn làm đáp án. `context/boundary` = mốc thời gian hoặc đoạn phụ để kiểm chứng câu hỏi."
            )

        if evidence:
            evidence_audio = make_span_masked_audio(mixture, sample_rate, evidence, evidence_pad)
            st.audio(wav_bytes(sample_rate, evidence_audio), format="audio/wav")
            st.caption(
                "Audio evidence ở đây là mixture được mask theo span dự đoán. Đây là prototype localization, chưa phải AudioSep/SAM-Audio separated waveform."
            )
        else:
            st.warning("Không có evidence span được chọn.")

    st.divider()
    st.caption(
        "Giới hạn hiện tại: hỏi mở hơn 4 lựa chọn, nhưng vẫn bị giới hạn bởi label taxonomy trong QCES cache/proposal head. Muốn giống AF3 thật cần thêm open-vocabulary event detector hoặc retrieval label bank lớn hơn."
    )


if __name__ == "__main__":
    main()
