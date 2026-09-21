#!/usr/bin/env python3
"""Standalone upload-and-separate demo for noisy in-car speech."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INFER_SCRIPT = PROJECT_ROOT / "code/mixi_understanding/scripts/infer_qces_uploaded_car_audio.py"
UPLOAD_ROOT = PROJECT_ROOT / "outputs/qces_uploaded_car_audio"
FRCRN_PYTHON = PROJECT_ROOT / ".venv-clearvoice/bin/python"
AUDIOSEP_PYTHON = Path("/home/cuongpv/anaconda3/envs/qces-sam/bin/python")


st.set_page_config(page_title="Upload Car Speech", page_icon="🚗", layout="wide")
st.title("UPLOAD AUDIO · Tách lời nói trong xe có nhiễu")
st.markdown(
    """
    Upload một file `WAV` hoặc `FLAC` có lời nói và tiếng xe. Demo sẽ chạy:

    `mixture → FRCRN / AudioSep → speech-enhanced audio`
    """
)
st.warning("FRCRN và AudioSep đều là model frozen. Output dùng để nghe/ASR, không phải bằng chứng rằng event detector đã nhận dạng đúng.")

uploaded = st.file_uploader("Chọn audio", type=["wav", "flac"])
method = st.radio("Model tách", ["FRCRN", "AudioSep", "Cả hai"], horizontal=True)
prompt = st.text_input("AudioSep prompt", value="a person speaking")

if uploaded is None:
    st.info("Upload một file WAV/FLAC để bắt đầu.")
    st.stop()

raw = uploaded.getvalue()
st.subheader("Input")

if st.button("Chạy tách speech", type="primary"):
    digest = hashlib.sha256(raw).hexdigest()[:16]
    run_dir = UPLOAD_ROOT / digest
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower() or ".wav"
    input_path = run_dir / f"input{suffix}"
    input_path.write_bytes(raw)
    requested = {"FRCRN": "frcrn", "AudioSep": "audiosep", "Cả hai": "both"}[method]
    methods = ("frcrn", "audiosep") if requested == "both" else (requested,)
    progress = st.empty()
    for current in methods:
        python = FRCRN_PYTHON if current == "frcrn" else AUDIOSEP_PYTHON
        output_path = run_dir / f"{current}.flac"
        command = [
            str(python),
            str(INFER_SCRIPT),
            "--method",
            current,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        if current == "audiosep":
            command.extend(["--prompt", prompt])
        progress.info(f"Đang chạy {current.upper()}...")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "code")
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            st.error(f"{current.upper()} failed")
            st.code((error.stderr or error.stdout or str(error))[-6000:])
            continue
        if output_path.is_file():
            st.subheader(f"{current.upper()} output")
            st.audio(str(output_path))
            st.caption(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")
            with st.expander(f"{current.upper()} log"):
                st.code((completed.stdout or "done")[-4000:])
    progress.success("Hoàn tất.")
