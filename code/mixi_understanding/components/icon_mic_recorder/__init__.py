"""Accessible icon-only microphone recorder for the chat composer."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


_BUILD_DIR = Path(__file__).resolve().parent / "frontend" / "build"
_component = components.declare_component("qces_icon_mic_recorder", path=str(_BUILD_DIR))


def icon_mic_recorder(
    *,
    appearance: str,
    start_label: str = "Bắt đầu ghi âm",
    stop_label: str = "Dừng ghi âm",
    audio_format: str = "wav",
    key: str,
) -> dict[str, Any] | None:
    """Render a 48px SVG microphone control and return newly recorded bytes."""

    last_id_key = f"{key}_last_audio_id"
    if last_id_key not in st.session_state:
        st.session_state[last_id_key] = 0
    value = _component(
        start_prompt=start_label,
        stop_prompt=stop_label,
        use_container_width=False,
        format=audio_format,
        appearance="light" if appearance == "Sáng" else "dark",
        key=key,
        default=None,
    )
    if value is None:
        return None
    audio_id = int(value["id"])
    if audio_id <= int(st.session_state[last_id_key]):
        return None
    st.session_state[last_id_key] = audio_id
    return {
        "bytes": base64.b64decode(value["audio_base64"]),
        "sample_rate": int(value["sample_rate"]),
        "sample_width": int(value["sample_width"]),
        "format": str(value["format"]),
        "id": audio_id,
    }
