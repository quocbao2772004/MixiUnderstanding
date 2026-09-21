#!/usr/bin/env python3
"""Polished, voice-first chat interface for the live QCES AudioQA stack."""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import streamlit as st
from PIL import Image, ImageDraw

from mixi_understanding.apps.qces_qwen_audioqa_demo import (
    COMBINED_ONTOLOGY,
    best_quote_span,
    boundary_relation_for_speech,
    group_live_inventory,
    live_inference_health,
    materialize_live_audio,
    parsed_program,
    post_live_inference,
    post_qwen,
    qwen_health,
    speaker_answer,
)
from mixi_understanding.components.icon_mic_recorder import icon_mic_recorder


APP_TITLE = "QCES"
EVIDENCE_SEPARATOR_URL = "http://127.0.0.1:8516"
OVERLAP_SEPARATOR_URL = "http://127.0.0.1:8517"
MULTISPEAKER_REGION_URL = "http://127.0.0.1:8518"
EVIDENCE_METHOD_NAMES = {
    "ours": "QCES · Tinh chỉnh phổ–thời gian",
    "audiosep": "QCES · Tách theo ngữ nghĩa",
}

CLASS_NAMES_VI = {
    "Accelerating_and_revving": "Tiếng động cơ tăng tốc/rồ ga",
    "Bark": "Tiếng chó sủa",
    "Howl": "Tiếng chó tru",
    "Whimper_(dog)": "Tiếng chó rên",
    "Meow": "Tiếng mèo kêu",
    "Medium_engine_(mid_frequency)": "Tiếng động cơ xe tần số trung",
    "Purr": "Tiếng mèo rừ",
    "Caterwaul": "Tiếng mèo gào",
    "Clapping": "Tiếng vỗ tay",
    "Laughter": "Tiếng cười",
    "Giggle": "Tiếng cười khúc khích",
    "Conversation": "Tiếng trò chuyện",
    "Shout": "Tiếng la hét",
    "Crying_and_sobbing": "Tiếng khóc và nức nở",
    "Knock": "Tiếng gõ",
    "Slam": "Tiếng đóng sầm",
    "Thump_and_thud": "Tiếng đập và tiếng thình thịch",
    "Traffic_noise_and_roadway_noise": "Tiếng giao thông và đường phố",
    "Heavy_engine_(low_frequency)": "Động cơ hạng nặng, tần số thấp",
    "Reversing_beeps": "Tiếng bíp lùi xe",
    "Speech": "Tiếng người nói",
    "Air_horn_and_truck_horn": "Còi hơi và còi xe tải",
    "Police_car_(siren)": "Còi xe cảnh sát",
    "Engine_starting": "Tiếng khởi động động cơ",
}

CLASS_GROUPS = (
    (
        "Động vật",
        ("Bark", "Howl", "Whimper_(dog)", "Meow", "Purr", "Caterwaul"),
    ),
    (
        "Con người",
        (
            "Speech", "Conversation", "Laughter", "Giggle", "Shout",
            "Crying_and_sobbing", "Clapping",
        ),
    ),
    ("Va chạm và tác động", ("Knock", "Slam", "Thump_and_thud")),
    (
        "Xe cộ và đường phố",
        (
            "Accelerating_and_revving", "Medium_engine_(mid_frequency)",
            "Traffic_noise_and_roadway_noise", "Heavy_engine_(low_frequency)",
            "Reversing_beeps", "Air_horn_and_truck_horn",
            "Police_car_(siren)", "Engine_starting",
        ),
    ),
)

EVIDENCE_LABEL_ALIASES = {
    # The public separator/refiner checkpoint predates the two detector leaves.
    # Route them to the nearest trained semantic condition until that evidence
    # model is expanded; do not pretend the refiner itself learned new labels.
    "Accelerating_and_revving": "Engine_starting",
    "Medium_engine_(mid_frequency)": "Heavy_engine_(low_frequency)",
}


def brand_favicon() -> Image.Image:
    """Return the purple QCES waveform mark for the browser tab."""

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, 62, 62), radius=18, fill=(124, 58, 237, 255))
    for center_x, height in ((16, 16), (24, 28), (32, 40), (40, 28), (48, 16)):
        draw.rounded_rectangle(
            (center_x - 2, 32 - height // 2, center_x + 2, 32 + height // 2),
            radius=2,
            fill=(255, 255, 255, 255),
        )
    return image


def inject_styles(theme: str) -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
          --q-bg: #090c14;
          --q-panel: #101522;
          --q-card: #151b2b;
          --q-card-2: #1a2235;
          --q-border: rgba(255, 255, 255, 0.09);
          --q-border-strong: rgba(139, 92, 246, 0.42);
          --q-text: #f8fafc;
          --q-muted: #9ca9bd;
          --q-purple: #8b5cf6;
          --q-purple-soft: rgba(139, 92, 246, 0.15);
          --q-cyan: #22d3ee;
          --q-success: #34d399;
          --q-warning: #fbbf24;
          --q-radius: 18px;
        }

        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        h1, h2, h3, .brand-name { font-family: 'Space Grotesk', sans-serif !important; }
        .stApp {
          background:
            radial-gradient(circle at 85% 8%, rgba(34, 211, 238, 0.075), transparent 28rem),
            radial-gradient(circle at 25% 0%, rgba(139, 92, 246, 0.11), transparent 30rem),
            var(--q-bg);
          color: var(--q-text);
        }
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }
        .block-container {
          max-width: 1040px;
          padding-top: 1.6rem;
          padding-bottom: 7.5rem;
        }
        [data-testid="stSidebar"] {
          background: rgba(15, 20, 32, 0.96);
          border-right: 1px solid var(--q-border);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
        [data-testid="stSidebar"] * { color: var(--q-text); }

        .brand-lockup { display:flex; gap:12px; align-items:center; margin: 0 0 1.45rem; }
        .brand-mark {
          width:42px; height:42px; border-radius:13px; display:grid; place-items:center;
          color:white; background:#7c3aed; border:1px solid rgba(255,255,255,.15);
          box-shadow:0 12px 32px rgba(124,58,237,.24);
        }
        .brand-name { font-weight:700; letter-spacing:-.02em; line-height:1.05; }
        .brand-sub { color:var(--q-muted); font-size:.76rem; margin-top:4px; }
        .topbar {
          display:flex; align-items:flex-start; justify-content:space-between; gap:18px;
          margin: .25rem 0 1.5rem;
        }
        .topbar h1 { font-size:clamp(1.55rem, 4vw, 2.3rem); margin:0; letter-spacing:-.045em; }
        .topbar p { color:var(--q-muted); margin:.42rem 0 0; font-size:.96rem; }
        .live-badge {
          flex:none; display:inline-flex; align-items:center; gap:8px; min-height:36px;
          padding:7px 12px; border:1px solid rgba(52,211,153,.25); border-radius:999px;
          background:rgba(52,211,153,.08); color:#a7f3d0; font-size:.78rem; font-weight:700;
        }
        .live-dot { width:7px; height:7px; border-radius:50%; background:var(--q-success); box-shadow:0 0 12px var(--q-success); }
        .offline-badge { border-color:rgba(248,113,113,.25); background:rgba(248,113,113,.08); color:#fecaca; }
        .offline-badge .live-dot { background:#f87171; box-shadow:0 0 12px #f87171; }

        .hero-shell {
          position:relative; overflow:hidden; border:1px solid var(--q-border-strong);
          border-radius:24px; padding:clamp(1.4rem, 4vw, 2.25rem);
          background:linear-gradient(145deg, rgba(139,92,246,.13), rgba(21,27,43,.92) 52%, rgba(34,211,238,.05));
          box-shadow:0 30px 80px rgba(0,0,0,.25); margin-bottom:1.35rem;
        }
        .hero-shell:after {
          content:""; position:absolute; width:210px; height:210px; right:-80px; top:-105px;
          border:1px solid rgba(34,211,238,.16); border-radius:50%; box-shadow:0 0 0 34px rgba(34,211,238,.025), 0 0 0 72px rgba(139,92,246,.02);
          pointer-events:none;
        }
        .eyebrow { color:#c4b5fd; font-size:.74rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
        .hero-shell h2 { margin:.65rem 0 .65rem; max-width:680px; font-size:clamp(1.5rem, 4vw, 2.45rem); letter-spacing:-.045em; line-height:1.08; }
        .hero-shell p { color:#b8c2d3; max-width:660px; line-height:1.65; margin:0; }
        .cap-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:1.25rem; }
        .cap-pill { border:1px solid var(--q-border); background:rgba(255,255,255,.035); color:#d9e1ec; padding:7px 10px; border-radius:999px; font-size:.76rem; }

        .context-card {
          border:1px solid var(--q-border); background:rgba(16,21,34,.9); border-radius:20px;
          padding:1rem 1.1rem; margin:0 0 1.1rem; box-shadow:0 16px 44px rgba(0,0,0,.16);
        }
        .context-head { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:.85rem; }
        .context-title { display:flex; align-items:center; gap:10px; font-weight:700; }
        .context-title svg { width:18px; height:18px; color:var(--q-cyan); }
        .context-meta { color:var(--q-muted); font-size:.78rem; }
        .timeline { position:relative; height:42px; border-radius:12px; background:#0a0f1a; border:1px solid var(--q-border); overflow:hidden; }
        .timeline-grid { position:absolute; inset:0; background:repeating-linear-gradient(90deg, transparent 0, transparent calc(20% - 1px), rgba(255,255,255,.045) 20%); }
        .event-segment { position:absolute; height:9px; border-radius:999px; min-width:4px; background:linear-gradient(90deg, var(--q-purple), var(--q-cyan)); box-shadow:0 0 16px rgba(34,211,238,.25); }
        .event-chips { display:flex; gap:7px; flex-wrap:wrap; margin-top:.8rem; }
        .event-chip { display:inline-flex; align-items:center; gap:7px; padding:6px 9px; border-radius:9px; background:var(--q-card-2); border:1px solid var(--q-border); color:#dce4ef; font-size:.75rem; }
        .event-chip i { width:6px; height:6px; border-radius:50%; background:var(--q-cyan); }
        .transcript-box { margin-top:.9rem; border-top:1px solid var(--q-border); padding-top:.85rem; color:#cbd5e1; line-height:1.55; font-size:.88rem; }
        .transcript-label { color:var(--q-muted); font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.3rem; }
        .speaker-line { display:grid; grid-template-columns:104px minmax(0,1fr); gap:10px; padding:.28rem 0; }
        .speaker-tag { color:#a5f3fc; font-size:.74rem; font-weight:700; }

        [data-testid="stChatMessage"] {
          background:rgba(16,21,34,.72); border:1px solid var(--q-border);
          border-radius:18px; padding:1rem 1.05rem; margin:.7rem 0;
          animation:message-in .28s ease-out both;
        }
        [data-testid="stChatMessage"] p { line-height:1.6; }
        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] { background:var(--q-purple); }
        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] { background:#0e7490; }
        .answer-kicker { color:#9ca9bd; text-transform:uppercase; letter-spacing:.1em; font-size:.68rem; font-weight:700; margin-bottom:.38rem; }
        .answer-main { color:#f8fafc; font-family:'Space Grotesk',sans-serif; font-size:1.12rem; font-weight:600; line-height:1.5; overflow-wrap:anywhere; }
        .answer-meta { display:flex; flex-wrap:wrap; gap:7px; margin-top:.75rem; }
        .meta-pill { display:inline-flex; align-items:center; min-height:27px; padding:4px 8px; border-radius:999px; font-size:.7rem; color:#cbd5e1; background:#182033; border:1px solid var(--q-border); }
        .evidence-label { margin-top:.9rem; color:#a5f3fc; font-size:.78rem; font-weight:700; }
        .evidence-caption { color:var(--q-muted); font-size:.76rem; margin-top:.35rem; }
        .evidence-method-card {
          border:1px solid var(--q-border); background:rgba(16,21,34,.72);
          border-radius:16px; padding:.8rem 1rem .25rem; margin:.85rem 0 1rem;
        }
        .evidence-method-title { color:var(--q-text); font-size:.82rem; font-weight:700; }
        .evidence-method-help { color:var(--q-muted); font-size:.74rem; line-height:1.45; margin:.2rem 0 .35rem; }
        .notice { border-left:3px solid var(--q-cyan); color:#b9c5d6; background:rgba(34,211,238,.05); padding:.75rem .9rem; border-radius:0 10px 10px 0; font-size:.82rem; line-height:1.55; margin:.75rem 0; }

        div[data-testid="stChatInput"] { background:rgba(9,12,20,.86); backdrop-filter:blur(18px); padding-top:.5rem; }
        div[data-testid="stChatInput"] > div { border:1px solid rgba(139,92,246,.38); background:#141a29; border-radius:16px; box-shadow:0 18px 50px rgba(0,0,0,.3); }
        div[data-testid="stChatInput"] textarea { color:var(--q-text); min-height:48px; }
        div[data-testid="stChatInput"] textarea:focus { box-shadow:0 0 0 3px rgba(139,92,246,.2); }

        .stButton > button, .stDownloadButton > button {
          min-height:44px; border-radius:12px; border:1px solid var(--q-border);
          background:#171e2f; color:#e8edf5; font-weight:600; cursor:pointer;
          transition:border-color .2s ease, background .2s ease, box-shadow .2s ease, transform .2s ease;
        }
        .stButton > button:hover { border-color:rgba(34,211,238,.45); background:#1c2639; transform:translateY(-1px); }
        .stButton > button:focus-visible { outline:3px solid rgba(139,92,246,.45); outline-offset:2px; }
        .stButton > button[kind="primary"] { background:#7c3aed; border-color:#8b5cf6; color:white; box-shadow:0 10px 25px rgba(124,58,237,.22); }
        .stButton > button[kind="primary"]:hover { background:#8b5cf6; border-color:#a78bfa; }
        [data-testid="stFileUploader"] { width:52px; height:52px; border:0!important; padding:0!important; background:transparent!important; }
        /* Keep the composer compact after a file is selected. The attachment
           remains in Streamlit state; only its redundant filename/delete row
           is hidden because the paperclip already provides the replace action. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
          display:none!important;
        }
        [data-testid="stFileUploaderDropzone"] { width:52px; min-width:52px; height:52px; min-height:52px; padding:2px!important; border:0!important; background:transparent!important; }
        [data-testid="stFileUploaderDropzoneInstructions"] { display:none!important; }
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span { display:none!important; }
        [data-testid="stFileUploaderDropzone"] button {
          position:relative; width:48px!important; min-width:48px!important; height:48px!important;
          min-height:48px!important; padding:0!important; border-radius:50%!important; overflow:hidden;
          font-size:0!important; line-height:0; white-space:nowrap; color:#ffffff!important;
          background:#111318!important; border:1px solid #30343b!important; box-shadow:none!important;
          cursor:pointer; touch-action:manipulation;
          transition:background-color .18s ease, color .18s ease;
        }
        [data-testid="stFileUploaderDropzone"] button:before {
          content:""; display:block; width:24px; height:24px; margin:auto; background:currentColor;
          -webkit-mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48'/%3E%3C/svg%3E") center/contain no-repeat;
          mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48'/%3E%3C/svg%3E") center/contain no-repeat;
        }
        [data-testid="stFileUploaderDropzone"] button:hover { color:#ffffff!important; background:#20242a!important; border-color:#59606b!important; }
        [data-testid="stFileUploaderDropzone"] button:active { background:#292e35!important; }
        [data-testid="stFileUploaderDropzone"] button:focus-visible { outline:3px solid rgba(22,138,255,.48); outline-offset:1px; }
        [data-testid="stTextInput"] { width:100%; }
        [data-testid="stTextInput"] [data-testid="textInputRootElement"] {
          min-height:52px; overflow:hidden; border-radius:999px!important;
          background:#111318!important; border:1px solid #30343b!important;
          box-shadow:none!important;
        }
        [data-testid="stTextInput"] [data-testid="textInputRootElement"]:focus-within {
          border-color:#168aff!important; box-shadow:0 0 0 3px rgba(22,138,255,.2)!important;
        }
        [data-testid="stTextInput"] input {
          min-height:52px; padding:0 20px; border-radius:999px; color:#ffffff;
          background:transparent!important; border:0!important; outline:0!important; box-shadow:none!important;
          font-size:1rem; caret-color:#168aff;
        }
        [data-testid="stTextInput"] input::placeholder { color:#aeb4bf; opacity:1; }
        [data-testid="stTextInput"] input:disabled {
          color:#aeb4bf; -webkit-text-fill-color:#aeb4bf; background:#2c2e33; opacity:1;
        }
        [data-testid="stTextInput"] input:focus {
          border-color:transparent!important; box-shadow:none!important;
        }
        iframe[title*="qces_icon_mic_recorder"] { width:52px!important; min-width:52px; height:52px!important; border:0; border-radius:50%; }
        [data-testid="stHorizontalBlock"]:has(iframe[title*="qces_icon_mic_recorder"]) {
          display:grid!important; grid-template-columns:52px 52px minmax(240px,1fr);
          gap:8px!important; align-items:center;
          padding:.35rem 0 .2rem;
        }
        [data-testid="stHorizontalBlock"]:has(iframe[title*="qces_icon_mic_recorder"]) > [data-testid="column"] {
          width:auto!important; min-width:0!important; flex:none!important;
        }
        [data-testid="stAudio"] { width:100%; border-radius:12px; overflow:hidden; }
        [data-testid="stAlert"] { border-radius:14px; border-color:var(--q-border); background:#151b2b; color:var(--q-text); }
        hr { border-color:var(--q-border); }
        .sidebar-label { color:#93a1b6; font-size:.69rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin:1rem 0 .5rem; }
        .model-row { display:flex; justify-content:space-between; gap:10px; align-items:center; padding:.65rem 0; border-bottom:1px solid var(--q-border); font-size:.8rem; }
        .model-row span:last-child { color:#86efac; font-size:.72rem; }
        .class-panel {
          padding:.8rem; border:1px solid var(--q-border); border-radius:14px;
          background:rgba(255,255,255,.025);
        }
        .class-summary { color:var(--q-text); font-size:.82rem; font-weight:700; margin-bottom:.8rem; }
        .class-summary span { color:var(--q-muted); font-size:.72rem; font-weight:500; margin-left:5px; }
        .class-group + .class-group { margin-top:.8rem; }
        .class-group-title { color:var(--q-muted); font-size:.66rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; margin-bottom:.4rem; }
        .class-list { display:flex; flex-wrap:wrap; gap:5px; }
        .class-chip {
          display:inline-flex; align-items:center; min-height:26px; padding:4px 7px;
          border-radius:8px; border:1px solid var(--q-border); background:#171e2f;
          color:#e8edf5; font-size:.68rem; line-height:1.25;
        }

        @keyframes message-in { from { opacity:0; transform:translateY(7px); } to { opacity:1; transform:none; } }
        @media (prefers-reduced-motion: reduce) {
          *, *:before, *:after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; }
        }
        @media (max-width: 768px) {
          .block-container { padding:1rem .85rem 7rem; }
          .topbar { align-items:flex-start; }
          .topbar p { font-size:.86rem; }
          .hero-shell { border-radius:18px; }
          .live-badge { padding:6px 9px; font-size:.7rem; }
          [data-testid="stChatMessage"] { padding:.85rem; }
          [data-testid="stHorizontalBlock"]:has(iframe[title*="qces_icon_mic_recorder"]) {
            grid-template-columns:48px 48px minmax(0,1fr); gap:6px!important;
          }
          [data-testid="stTextInput"] input { min-height:48px; padding:0 16px; }
          iframe[title*="qces_icon_mic_recorder"] { width:48px!important; min-width:48px; height:52px!important; }
          [data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"] { width:48px; min-width:48px; }
        }
        @media (max-width: 375px) {
          .block-container { padding-left:.65rem; padding-right:.65rem; }
          .topbar { gap:10px; }
          .topbar h1 { font-size:1.45rem; }
          .live-badge { max-width:106px; line-height:1.25; }
          .context-card, [data-testid="stChatMessage"] { border-radius:14px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if theme == "Sáng":
        st.markdown(
            """
            <style>
            :root {
              --q-bg:#f8f7ff; --q-panel:#ffffff; --q-card:#ffffff; --q-card-2:#f4f1ff;
              --q-border:#e4e0f2; --q-border-strong:#c4b5fd; --q-text:#1e1b4b;
              --q-muted:#5f687d; --q-purple:#6d28d9; --q-purple-soft:#ede9fe;
              --q-cyan:#0e7490; --q-success:#047857; --q-warning:#a16207;
            }
            .stApp {
              background:
                radial-gradient(circle at 84% 2%, rgba(8,145,178,.10), transparent 28rem),
                radial-gradient(circle at 18% 0%, rgba(124,58,237,.11), transparent 30rem),
                var(--q-bg);
              color:var(--q-text);
              color-scheme:light;
            }
            header[data-testid="stHeader"] { background:rgba(248,247,255,.82); }
            [data-testid="stSidebar"] { background:rgba(255,255,255,.96); border-right:1px solid var(--q-border); }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
            .stApp p, .stApp label, .stApp strong, .stApp li,
            .stApp [data-testid="stMarkdownContainer"],
            .stApp [data-testid="stMarkdownContainer"] p,
            .stApp [data-testid="stMarkdownContainer"] strong,
            .stApp [data-testid="stText"],
            [data-testid="stSidebar"] * { color:var(--q-text) !important; }
            .topbar h1 { color:#1e1b4b !important; }
            .brand-mark { box-shadow:0 12px 30px rgba(109,40,217,.18); }
            .brand-sub, .topbar p, .context-meta, .evidence-caption, .transcript-label,
            .sidebar-label { color:var(--q-muted) !important; }
            .live-badge { color:#065f46; background:#ecfdf5; border-color:#a7f3d0; }
            .offline-badge { color:#991b1b; background:#fef2f2; border-color:#fecaca; }
            .hero-shell {
              background:linear-gradient(145deg,#ffffff 16%,#f5f3ff 62%,#ecfeff);
              box-shadow:0 24px 70px rgba(51,36,99,.10); border-color:#d8ccff;
            }
            .eyebrow { color:#6d28d9; }
            .hero-shell h2, .hero-shell p { color:#29234e !important; }
            .cap-pill { background:#ffffff; border-color:#ddd6fe; color:#473f68; }
            .context-card, [data-testid="stChatMessage"] {
              background:rgba(255,255,255,.93); border-color:var(--q-border);
              box-shadow:0 14px 40px rgba(51,36,99,.075);
            }
            .timeline { background:#f4f1ff; border-color:#ddd6fe; }
            .timeline-grid { background:repeating-linear-gradient(90deg,transparent 0,transparent calc(20% - 1px),rgba(76,29,149,.08) 20%); }
            .event-chip, .meta-pill { background:#f5f3ff; border-color:#ddd6fe; color:#40385f; }
            .transcript-box { color:#312e57; border-color:#e4e0f2; }
            .speaker-tag { color:#0e7490; }
            .answer-kicker { color:#6b647d; }
            .answer-main { color:#1e1b4b; }
            .evidence-label { color:#0e7490; }
            .evidence-method-card { background:#ffffff; border-color:#e4e0f2; }
            .evidence-method-title { color:#1e1b4b; }
            .evidence-method-help { color:#5f687d; }
            .notice { color:#334155; background:#ecfeff; border-left-color:#0891b2; }
            .stButton > button, .stDownloadButton > button {
              background:#ffffff; color:#30294f; border-color:#d8d1eb;
              box-shadow:0 4px 14px rgba(51,36,99,.06);
            }
            .stButton > button:hover { background:#f5f3ff; border-color:#a78bfa; }
            .stButton > button[kind="primary"] { background:#6d28d9; color:#fff; border-color:#6d28d9; }
            .stButton > button p { color:inherit !important; }
            .stButton > button[kind="primary"]:hover { background:#7c3aed; border-color:#7c3aed; }
            [data-testid="stFileUploader"] { background:transparent!important; border:0!important; }
            [data-testid="stFileUploaderDropzone"] { background:transparent!important; border:0!important; color:#111827; }
            [data-testid="stFileUploaderDropzone"] button { color:#111827!important; background:#ffffff!important; border:1px solid #d9dde5!important; box-shadow:none!important; }
            [data-testid="stFileUploaderDropzone"] button:hover { color:#000000!important; background:#f3f4f6!important; border-color:#b8bec9!important; }
            [data-testid="stFileUploaderDropzone"] button:active { background:#e5e7eb!important; }
            [data-testid="stFileUploaderDropzone"] span,
            [data-testid="stFileUploaderDropzone"] div,
            [data-testid="stFileUploaderDropzone"] small { color:#30294f !important; }
            [data-testid="stAlert"] { background:#fff; color:#30294f; border-color:#ddd6fe; }
            [data-testid="stAlert"] * { color:#30294f !important; }
            [data-testid="stTextInput"] input {
              min-height:52px; border-radius:999px; color:#111827; background:transparent!important;
              border:0!important; box-shadow:none!important;
            }
            [data-testid="stTextInput"] [data-testid="textInputRootElement"] {
              background:#ffffff!important; border:1px solid #d9dde5!important; box-shadow:none!important;
            }
            [data-testid="stTextInput"] [data-testid="textInputRootElement"]:focus-within {
              border-color:#0866ff!important; box-shadow:0 0 0 3px rgba(8,102,255,.16)!important;
            }
            [data-testid="stTextInput"] input::placeholder { color:#475569; opacity:1; }
            [data-testid="stTextInput"] input:disabled { color:#475569; -webkit-text-fill-color:#475569; background:transparent!important; opacity:1; }
            [data-testid="stTextInput"] input:focus { border-color:transparent!important; box-shadow:none!important; }
            hr { border-color:#e4e0f2; }
            .model-row { border-color:#ebe7f4; }
            .model-row span:last-child { color:#047857; }
            .class-panel { background:#ffffff; border-color:#e4e0f2; }
            .class-summary { color:#1e1b4b; }
            .class-summary span, .class-group-title { color:#5f687d; }
            .class-chip { background:#f5f3ff; border-color:#ddd6fe; color:#30294f; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Keep the composer deterministic. Streamlit's BaseWeb input applies its own
    # disabled/theme colors, so this final theme-specific block must be injected
    # after every shared/light override above.
    composer_bg = "#ffffff" if theme == "Sáng" else "#111318"
    composer_fg = "#000000" if theme == "Sáng" else "#ffffff"
    composer_border = "#d9dde5" if theme == "Sáng" else "#30343b"
    composer_hover = "#f3f4f6" if theme == "Sáng" else "#20242a"
    composer_shadow = (
        "0 8px 24px rgba(15,23,42,.08)"
        if theme == "Sáng"
        else "0 8px 24px rgba(0,0,0,.28)"
    )
    st.markdown(
        f"""
        <style>
        [data-testid="stHorizontalBlock"]:has(iframe[title*="qces_icon_mic_recorder"]) {{
          background:{composer_bg}!important;
          border:1px solid {composer_border}!important;
          border-radius:999px!important;
          box-shadow:{composer_shadow}!important;
          padding:7px!important;
        }}
        [data-testid="stTextInput"],
        [data-testid="stTextInput"] > div,
        [data-testid="stTextInput"] [data-testid="textInputRootElement"],
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stTextInput"] [data-baseweb="base-input"] {{
          background:{composer_bg}!important;
          color:{composer_fg}!important;
          box-shadow:none!important;
        }}
        [data-testid="stTextInput"] [data-testid="textInputRootElement"],
        [data-testid="stTextInput"] [data-baseweb="input"] {{
          border:1px solid {composer_border}!important;
        }}
        [data-testid="stTextInput"] input,
        [data-testid="stTextInput"] input:disabled {{
          color:{composer_fg}!important;
          -webkit-text-fill-color:{composer_fg}!important;
          background:{composer_bg}!important;
          background-color:{composer_bg}!important;
          opacity:1!important;
        }}
        [data-testid="stTextInput"] input::placeholder {{
          color:{composer_fg}!important;
          -webkit-text-fill-color:{composer_fg}!important;
          opacity:1!important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
          color:{composer_fg}!important;
          background:{composer_bg}!important;
          border:1px solid {composer_border}!important;
          box-shadow:none!important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
          color:{composer_fg}!important;
          background:{composer_hover}!important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def svg_icon(name: str) -> str:
    icons = {
        "wave": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 10v4M8 7v10M12 3v18M16 7v10M20 10v4"/></svg>',
        "audio": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>',
    }
    return icons[name]


def audio_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(0.01, float(result.stdout.strip()))


def evidence_separator_health() -> bool:
    try:
        with urllib.request.urlopen(EVIDENCE_SEPARATOR_URL + "/health", timeout=0.8) as response:
            return bool(json.loads(response.read().decode("utf-8")).get("ok"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def overlap_separator_health() -> bool:
    try:
        with urllib.request.urlopen(OVERLAP_SEPARATOR_URL + "/health", timeout=0.8) as response:
            return bool(json.loads(response.read().decode("utf-8")).get("ok"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def multispeaker_region_health() -> bool:
    try:
        with urllib.request.urlopen(
            MULTISPEAKER_REGION_URL + "/health", timeout=0.8
        ) as response:
            return bool(json.loads(response.read().decode("utf-8")).get("ok"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


@st.cache_data(show_spinner=False)
def post_overlap_separation(audio_path_text: str) -> dict[str, Any]:
    request = urllib.request.Request(
        OVERLAP_SEPARATOR_URL + "/separate",
        data=json.dumps(
            {"audio_path": str(Path(audio_path_text).resolve())},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "error" in result:
        raise RuntimeError(f"{result['error']}: {result.get('message', '')}")
    return result


@st.cache_data(show_spinner=False)
def post_multispeaker_region_separation(
    audio_path_text: str,
    prediction_json: str,
) -> dict[str, Any]:
    request = urllib.request.Request(
        MULTISPEAKER_REGION_URL + "/separate",
        data=json.dumps(
            {
                "audio_path": str(Path(audio_path_text).resolve()),
                "prediction": json.loads(prediction_json),
                "max_speakers": 4,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "error" in result:
        raise RuntimeError(f"{result['error']}: {result.get('message', '')}")
    return result


@st.cache_data(show_spinner=False)
def post_evidence_separation(
    audio_path_text: str,
    start_seconds: float,
    end_seconds: float,
    labels: tuple[str, ...],
    method: str,
) -> dict[str, Any]:
    request = urllib.request.Request(
        EVIDENCE_SEPARATOR_URL + "/separate",
        data=json.dumps(
            {
                "audio_path": str(Path(audio_path_text).resolve()),
                "start_seconds": float(start_seconds),
                "end_seconds": float(end_seconds),
                "labels": list(labels),
                "method": method,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "error" in result:
        raise RuntimeError(f"{result['error']}: {result.get('message', '')}")
    return result


def render_brand() -> None:
    st.markdown(
        f"""
        <div class="brand-lockup">
          <div class="brand-mark">{svg_icon('wave')}</div>
          <div><div class="brand-name">QCES</div><div class="brand-sub">AudioQA có bằng chứng</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_service_rows(
    qwen_ok: bool,
    inference_ok: bool,
    evidence_ok: bool,
    overlap_ok: bool,
    multispeaker_ok: bool,
) -> None:
    st.markdown('<div class="sidebar-label">Hệ thống</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="model-row"><span>Hiểu câu hỏi</span><span>{'Sẵn sàng' if qwen_ok else 'Ngoại tuyến'}</span></div>
        <div class="model-row"><span>Nhận diện sự kiện</span><span>{'Sẵn sàng' if inference_ok else 'Ngoại tuyến'}</span></div>
        <div class="model-row"><span>Nhận dạng tiếng nói</span><span>{'Sẵn sàng' if inference_ok else 'Ngoại tuyến'}</span></div>
        <div class="model-row"><span>Phân biệt người nói</span><span>{'Sẵn sàng' if inference_ok else 'Ngoại tuyến'}</span></div>
        <div class="model-row"><span>Phân cụm nhiều người nói</span><span>{'Sẵn sàng' if multispeaker_ok else 'Ngoại tuyến'}</span></div>
        <div class="model-row"><span>Tách cục bộ giọng chồng lấn</span><span>{'Sẵn sàng' if overlap_ok else 'Ngoại tuyến'}</span></div>
        <div class="model-row"><span>Tách evidence</span><span>{'Sẵn sàng' if evidence_ok else 'Ngoại tuyến'}</span></div>
        """,
        unsafe_allow_html=True,
    )


def render_supported_classes() -> None:
    groups = []
    for group_name, class_keys in CLASS_GROUPS:
        chips = "".join(
            f'<span class="class-chip">{html.escape(CLASS_NAMES_VI[key])}</span>'
            for key in class_keys
        )
        groups.append(
            '<div class="class-group">'
            f'<div class="class-group-title">{html.escape(group_name)}</div>'
            f'<div class="class-list">{chips}</div>'
            "</div>"
        )
    st.markdown(
        '<div class="sidebar-label">Khả năng nhận diện</div>'
        '<div class="class-panel">'
        f'<div class="class-summary">{len(CLASS_NAMES_VI)} lớp âm thanh'
        '<span>đang hỗ trợ</span></div>'
        + "".join(groups)
        + "</div>",
        unsafe_allow_html=True,
    )


def queue_composer_question() -> None:
    question = str(st.session_state.get("audio_chat_question") or "").strip()
    if question:
        st.session_state.audio_chat_submit = question
        st.session_state.audio_chat_question = ""


def render_composer(has_audio: bool, inference_ok: bool) -> tuple[bytes | None, str, str]:
    mic_column, upload_column, input_column = st.columns([0.65, 0.65, 8.7])
    widget_version = int(st.session_state.audio_chat_source_widget_version)
    with mic_column:
        recorded = icon_mic_recorder(
            appearance=str(st.session_state.audio_chat_theme),
            start_label="Bắt đầu ghi âm",
            stop_label="Dừng ghi âm",
            audio_format="wav",
            key=f"audio_chat_microphone_{widget_version}",
        )
    with upload_column:
        uploaded = st.file_uploader(
            "Đính kèm tệp âm thanh",
            type=["wav", "flac", "mp3", "m4a", "ogg", "webm"],
            key=f"audio_chat_upload_{widget_version}",
            label_visibility="collapsed",
            disabled=not inference_ok,
        )
    with input_column:
        st.text_input(
            "Câu hỏi",
            key="audio_chat_question",
            placeholder=(
                "Hỏi về lời nói hoặc sự kiện trong audio…"
                if has_audio
                else "Thêm audio để bắt đầu…"
            ),
            label_visibility="collapsed",
            disabled=not has_audio,
            on_change=queue_composer_question,
        )
    if uploaded is not None:
        return uploaded.getvalue(), Path(uploaded.name).suffix or ".wav", uploaded.name
    if recorded is not None and recorded.get("bytes"):
        suffix = "." + str(recorded.get("format") or "wav").lstrip(".")
        return bytes(recorded["bytes"]), suffix, "Bản ghi từ microphone"
    return None, ".wav", ""


def speaker_display(value: Any) -> str:
    raw = str(value or "SPEAKER_01")
    suffix = raw.rsplit("_", 1)[-1].lstrip("0") or "1"
    return f"Người nói {suffix}"


def render_timeline(prediction: Mapping[str, Any], duration: float) -> None:
    events = sorted(
        list(prediction.get("predicted_events", [])),
        key=lambda row: float(row.get("start_seconds", 0.0)),
    )
    rows: list[str] = []
    for index, event in enumerate(events):
        start = max(0.0, float(event.get("start_seconds", 0.0)))
        end = min(duration, float(event.get("end_seconds", start)))
        left = min(99.0, 100.0 * start / duration)
        width = max(0.6, 100.0 * max(0.01, end - start) / duration)
        top = 8 + (index % 3) * 10
        label = html.escape(str(event.get("display_label") or event.get("label") or "Event"))
        rows.append(
            f'<span class="event-segment" title="{label}: {start:.2f}–{end:.2f}s" '
            f'style="left:{left:.3f}%;width:{min(width, 100.0-left):.3f}%;top:{top}px"></span>'
        )
    chips = "".join(
        '<span class="event-chip"><i></i>'
        + html.escape(str(event.get("display_label") or event.get("label") or "Event"))
        + f' · {float(event.get("start_seconds", 0.0)):.1f}s</span>'
        for event in events[:12]
    )
    separated_speakers = list(prediction.get("separated_speakers", []))
    transcript_source = separated_speakers or list(prediction.get("speech_segments", []))
    speech_rows = [
        segment for segment in transcript_source
        if str(segment.get("hypothesis") or "").strip()
    ]
    hypothesis = " ".join(
        str(segment.get("hypothesis") or "").strip() for segment in speech_rows
    ).strip()
    transcript = ""
    if hypothesis:
        speaker_count = int(prediction.get("speaker_count") or 1)
        if speaker_count > 1:
            transcript_body = "".join(
                '<div class="speaker-line"><span class="speaker-tag">'
                + html.escape(speaker_display(segment.get("speaker_id")))
                + '</span><span>'
                + html.escape(str(segment.get("hypothesis") or "").strip())
                + "</span></div>"
                for segment in speech_rows
            )
        else:
            transcript_body = html.escape(hypothesis)
        transcript_title = (
            "Bản chép lời sau tách giọng"
            if separated_speakers
            else "Bản chép lời trên audio hỗn hợp · chưa tách giọng"
        )
        transcript = (
            '<div class="transcript-box"><div class="transcript-label">'
            + transcript_title
            + "</div>"
            + transcript_body
            + "</div>"
        )
    speaker_count = int(prediction.get("speaker_count") or 0)
    speaker_meta = f" · {speaker_count} người nói" if speaker_count else ""
    st.markdown(
        f"""
        <div class="context-card">
          <div class="context-head">
            <div class="context-title">{svg_icon('audio')} Ngữ cảnh audio</div>
            <div class="context-meta">{duration:.1f} giây · {len(events)} sự kiện dự đoán{speaker_meta}</div>
          </div>
          <div class="timeline" aria-label="Timeline các sự kiện dự đoán"><span class="timeline-grid"></span>{''.join(rows)}</div>
          <div class="event-chips">{chips or '<span class="event-chip">Chưa phát hiện sự kiện rõ ràng</span>'}</div>
          {transcript}
        </div>
        """,
        unsafe_allow_html=True,
    )


def evidence_spans_for_speech(segments: list[Mapping[str, Any]]) -> list[tuple[float, float]]:
    return [
        (float(row["start_seconds"]), float(row["end_seconds"]))
        for row in segments
        if float(row.get("end_seconds", 0.0)) > float(row.get("start_seconds", 0.0))
    ]


def merge_evidence_spans(
    spans: list[tuple[float, float]], max_gap_seconds: float = 0.05,
) -> list[tuple[float, float]]:
    """Collapse overlapping event rows into stable listening windows."""
    merged: list[list[float]] = []
    for start, end in sorted(spans, key=lambda span: (span[0], span[1])):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + max_gap_seconds:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def naturalize_verified_answer(question: str, result: dict[str, Any]) -> dict[str, Any]:
    """Use the language model as a surface realizer, never as a new fact source."""

    verified_answer = str(result.get("answer") or "").strip()
    intent = str(result.get("intent") or "unknown")
    if not verified_answer or intent in {"error", "unsupported", "open_audioqa"}:
        return result
    response = post_qwen(
        "/verbalize",
        {
            "question": question,
            "verified_answer": verified_answer,
            "intent": intent,
            "evidence_spans": [list(span) for span in result.get("evidence", [])],
        },
        timeout=120,
    )
    candidate = str(response.get("answer") or "").strip()
    # The service is instructed to preserve the verified payload verbatim.  This
    # check makes the constraint executable and blocks a fluent hallucination.
    lowered_candidate = candidate.casefold()
    bare_answer = candidate.strip(" .,!?:;\"'").casefold() == verified_answer.strip(
        " .,!?:;\"'"
    ).casefold()
    persona_present = "mình" in lowered_candidate or "bạn" in lowered_candidate
    meta_opening = any(
        phrase in lowered_candidate
        for phrase in ("đây là câu trả lời", "câu trả lời của mình", "câu trả lời là")
    )
    templates = {
        "list": f"Trong audio của bạn, mình nghe thấy {verified_answer}.",
        "speaker_identity": f"Mình xác định giọng nói trong audio của bạn là {verified_answer}.",
        "speaker_count": f"Mình xác định trong audio của bạn có {verified_answer}.",
        "speech_content": f"Trong audio của bạn, mình nghe được nội dung: {verified_answer}.",
        "before": verified_answer,
        "after": verified_answer,
        "between": f"Mình nghe thấy {verified_answer} nằm giữa hai sự kiện bạn hỏi.",
        "first": verified_answer,
        "last": f"Sự kiện cuối cùng mình nghe thấy trong audio của bạn là {verified_answer}.",
        "count": f"Theo audio của bạn, mình đếm được {verified_answer}.",
        "exists": f"Dựa trên audio của bạn, mình xác định câu trả lời là {verified_answer}.",
        "locate": f"Trong audio của bạn, mình xác định vị trí là {verified_answer}.",
    }
    if (
        verified_answer.casefold() not in lowered_candidate
        or bare_answer
        or not persona_present
        or meta_opening
    ):
        candidate = templates.get(
            intent,
            f"Dựa trên audio của bạn, câu trả lời mình xác định được là {verified_answer}.",
        )
        result["naturalization_fallback"] = True
    else:
        result["naturalization_fallback"] = False
    result["structured_answer"] = verified_answer
    result["answer"] = candidate
    return result


def heuristic_question_program(question: str) -> dict[str, Any]:
    """Small local fallback for temporal Vietnamese queries if Qwen times out."""
    text = question.casefold()
    labels: list[str] = []
    if any(term in text for term in ("rồ ga", "ro ga", "tăng tốc", "ga xe")):
        labels.append("Accelerating_and_revving")
    if "khởi động" in text or "đề máy" in text:
        labels.append("Engine_starting")
    elif "động cơ xe" in text or "động cơ" in text:
        labels.append("Medium_engine_(mid_frequency)")
    intent = "after" if "sau" in text and "trước" not in text else "before"
    return {
        "intent": intent,
        "labels": labels[:2],
        "ordinals": [1 for _ in labels[:2]],
        "speaker": None,
        "quote": None,
        "confidence": 0.75,
        "reason": "local_temporal_fallback",
    }


def answer_question(question: str, audio_path: Path, prediction: Mapping[str, Any]) -> dict[str, Any]:
    try:
        parsed = post_qwen(
            "/parse",
            {"question": question, "labels": list(COMBINED_ONTOLOGY)},
        )
    except (OSError, RuntimeError, urllib.error.URLError):
        parsed = {"program": heuristic_question_program(question)}
    program = parsed_program(parsed)
    events = list(prediction.get("predicted_events", []))
    inventory = group_live_inventory(events)
    separated_speakers = list(prediction.get("separated_speakers", []))
    speech_source = separated_speakers or list(prediction.get("speech_segments", []))
    speech_segments = sorted(
        speech_source,
        key=lambda row: float(row.get("start_seconds", 0.0)),
    )
    requested_speaker = program.speaker or "any"
    selected_speech = [
        row for row in speech_segments
        if requested_speaker == "any"
        or str(row.get("predicted_speaker") or "unknown") == requested_speaker
    ]
    if program.intent == "speech_content" and program.ordinals:
        requested_speaker_id = f"SPEAKER_{int(program.ordinals[0]):02d}"
        selected_speech = [
            row for row in speech_segments
            if str(row.get("speaker_id") or "") == requested_speaker_id
        ]
    base: dict[str, Any] = {
        "intent": program.intent,
        "confidence": float(program.confidence),
        "reason": program.reason,
        "evidence": [],
        "evidence_labels": [],
        "note": "Evidence được suy ra từ prediction; audio live không có ground truth để tự chấm đúng/sai.",
    }

    def finalize() -> dict[str, Any]:
        return naturalize_verified_answer(question, base)

    if program.intent == "speaker_identity":
        speaker_count = int(prediction.get("speaker_count") or 0)
        if speaker_count > 1:
            genders = {
                speaker_answer(str(row.get("predicted_speaker") or "unknown"))
                for row in speech_segments
            }
            base["answer"] = f"{speaker_count} người nói; giọng nhận diện: {', '.join(sorted(genders))}"
        else:
            base["answer"] = speaker_answer(str(prediction.get("predicted_speaker") or "unknown"))
        base["evidence"] = evidence_spans_for_speech(speech_segments)
        base["evidence_labels"] = ["Speech"] if base["evidence"] else []
        return finalize()

    if program.intent == "speaker_count":
        speaker_count = int(prediction.get("speaker_count") or 0)
        base["answer"] = f"{speaker_count} người nói"
        base["evidence"] = evidence_spans_for_speech(speech_segments)
        base["evidence_labels"] = ["Speech"] if base["evidence"] else []
        return finalize()

    if program.intent == "speech_content":
        transcript_answer = (
            " ".join(
                str(row.get("hypothesis") or "").strip()
                for row in selected_speech
                if str(row.get("hypothesis") or "").strip()
            ).strip()
            or "Không phát hiện được lời nói phù hợp."
        )
        base["answer"] = transcript_answer
        base["evidence"] = evidence_spans_for_speech(selected_speech)
        base["evidence_labels"] = ["Speech"] if base["evidence"] else []
        base["direct_evidence_audio_paths"] = [
            str(row.get("audio_path")) for row in selected_speech
            if str(row.get("audio_path") or "").strip()
        ]
        return finalize()

    if program.intent == "list":
        # A list question is a scene summary, not a dump of detector rows.
        # Collapse per-turn Speech events into the estimated number of people
        # so labels such as “Speech · Người nói 1” never leak into the answer.
        speech_events = [
            event
            for event in events
            if str(event.get("label") or "") in {"Speech", "Conversation"}
        ]
        speaker_count = int(prediction.get("speaker_count") or 0)
        if speaker_count <= 0 and speech_events:
            speaker_ids = {
                str(event.get("speaker_id") or "")
                for event in speech_events
                if str(event.get("speaker_id") or "").strip()
            }
            speaker_count = len(speaker_ids)
        labels: list[str] = []
        if speech_events:
            labels.append(
                f"âm thanh của {speaker_count} người nói"
                if speaker_count
                else "âm thanh lời nói"
            )
        for event in events:
            label = str(event.get("label") or "")
            if label in {"Speech", "Conversation"}:
                continue
            display = CLASS_NAMES_VI.get(label) or str(event.get("display_label") or label)
            if display and display not in labels:
                labels.append(display)
        if labels:
            tail = " và ".join(
                value[:1].lower() + value[1:] if value else value
                for value in labels[1:]
            )
            base["answer"] = f"{labels[0]} cùng với {tail}" if tail else labels[0]
        else:
            base["answer"] = "Không phát hiện được sự kiện rõ ràng."
        base["evidence"] = merge_evidence_spans([
            (
                float(event.get("start_seconds", 0.0)),
                float(event.get("end_seconds", 0.0)),
            )
            for event in events
            if float(event.get("end_seconds", 0.0)) > float(event.get("start_seconds", 0.0))
        ])
        base["evidence_labels"] = list(
            dict.fromkeys(
                str(event.get("label") or "")
                for event in events
                if str(event.get("label") or "") in COMBINED_ONTOLOGY
            )
        )
        return finalize()

    if program.intent in {"speech_before_quote", "speech_after_quote"}:
        words = sorted(
            [word for row in selected_speech for word in row.get("words", [])],
            key=lambda row: float(row.get("start_seconds", 0.0)),
        )
        if not program.quote:
            base["answer"] = "Mình chưa xác định được câu trích dẫn trong câu hỏi."
            return finalize()
        match = best_quote_span(program.quote, words)
        if match is None or match[2] < 0.45:
            base["answer"] = "Mình chưa ground được câu trích dẫn vào transcript."
            return finalize()
        start, end, score = match
        before = program.intent == "speech_before_quote"
        selected_words = words[:start] if before else words[end:]
        answer = " ".join(str(word.get("text") or "").strip() for word in selected_words).strip()
        evidence_words = [*selected_words, *words[start:end]] if before else [*words[start:end], *selected_words]
        base["answer"] = answer or "Không có phần lời nói tương ứng."
        base["confidence"] = min(float(program.confidence), float(score))
        if evidence_words:
            base["evidence"] = [
                (
                    max(0.0, float(evidence_words[0]["start_seconds"]) - 0.08),
                    float(evidence_words[-1]["end_seconds"]) + 0.08,
                )
            ]
            base["evidence_labels"] = ["Speech"]
        return finalize()

    if program.intent == "open_audioqa":
        response = post_qwen(
            "/audio-answer",
            {"question": question, "audio_path": str(audio_path)},
        )
        base["answer"] = str(response.get("answer") or "Không có câu trả lời.")
        base["note"] = "Nhánh trả lời mở chưa có bộ kiểm chứng evidence độc lập."
        return base

    if program.intent == "unsupported":
        base["answer"] = "Câu này chưa phải một câu hỏi về nội dung audio mà hệ thống có thể xử lý."
        return base

    first_query = any(
        phrase in question.casefold()
        for phrase in ("tiếng nào xuất hiện trước", "âm thanh nào xuất hiện trước", "xuất hiện đầu tiên")
    )
    if first_query and (not program.labels or set(program.labels) <= {"Speech", "Conversation"}):
        first_events = [
            occurrence
            | {
                "label": item.get("label"),
                "display_label": item.get("display_label") or item.get("label"),
            }
            for item in inventory
            if str(item.get("label")) not in {"Speech", "Conversation"}
            for occurrence in item.get("occurrences", [])
        ]
        if first_events:
            first_event = min(first_events, key=lambda row: float(row.get("start_seconds", 0.0)))
            base["intent"] = "first"
            base["answer"] = f"{first_event['display_label']} xuất hiện đầu tiên."
            base["evidence"] = [
                (
                    float(first_event["start_seconds"]),
                    float(first_event["end_seconds"]),
                )
            ]
            base["evidence_labels"] = [str(first_event.get("label") or "")]
            return finalize()

    # For questions such as “which of A and B appears first?”, Qwen may use
    # the before/after intent with two labels. Compare those two requested
    # occurrences directly instead of treating the first label as a lone
    # anchor and looking for an unrelated neighbouring event.
    comparison_words = ("xuất hiện trước", "nào trước", "sớm hơn", "trước hơn")
    comparison_requested = any(word in question.casefold() for word in comparison_words)
    if (
        len(program.labels) >= 2
        and (program.intent in {"before", "after"} or comparison_requested)
    ):
        compared: list[dict[str, Any]] = []
        for index, label in enumerate(program.labels[:2]):
            item = next((row for row in inventory if str(row.get("label")) == label), None)
            occurrences = list(item.get("occurrences", [])) if item else []
            ordinal = program.ordinals[index] if index < len(program.ordinals) else 1
            # Qwen sometimes emits slot ordinals [1, 2] for two distinct
            # labels; ordinals are per-label, so a singleton second label is
            # still its first occurrence.
            if ordinal <= 0 or ordinal > len(occurrences):
                ordinal = 1
            if 0 < ordinal <= len(occurrences):
                occurrence = dict(occurrences[ordinal - 1])
                occurrence["label"] = label
                occurrence["display_label"] = str(item.get("display_label") or label)
                compared.append(occurrence)
        if len(compared) == 2:
            compared.sort(key=lambda row: float(row["start_seconds"]))
            chosen = compared[0] if program.intent != "after" else compared[-1]
            other = compared[-1] if program.intent != "after" else compared[0]
            relation = "sau" if program.intent == "after" else "trước"
            base["answer"] = (
                f"{chosen['display_label']} xuất hiện {relation} "
                f"{other['display_label']}."
            )
            base["evidence"] = [
                (
                    float(row["start_seconds"]),
                    float(row["end_seconds"]),
                )
                for row in compared
            ]
            base["evidence_labels"] = [str(row["label"]) for row in compared]
            if comparison_requested and program.intent == "between":
                base["intent"] = "before"
            return finalize()

    result = boundary_relation_for_speech(program, inventory)
    if not result.supported:
        base["answer"] = "Mình hiểu ý câu hỏi nhưng chưa ground được event anchor vào audio."
        return finalize()
    base["answer"] = result.answer if result.answer != "NONE" else "Không có sự kiện phù hợp."
    if result.evidence_window is not None:
        base["evidence"] = [result.evidence_window]
        base["evidence_labels"] = list(
            dict.fromkeys(
                str(event.get("label"))
                for event in result.evidence
                if str(event.get("label") or "") in COMBINED_ONTOLOGY
            )
        )
    return finalize()


def render_audio_download(path: Path, label: str, key: str) -> None:
    try:
        audio_bytes = path.read_bytes()
    except OSError:
        st.caption("Không đọc được file audio để tải xuống.")
        return
    st.download_button(
        label,
        data=audio_bytes,
        file_name=path.name,
        mime="audio/wav",
        key=key,
        use_container_width=True,
    )


def render_assistant_result(
    result: Mapping[str, Any], audio_path: Path, evidence_service_ok: bool,
) -> None:
    st.markdown(
        '<div class="answer-kicker">Câu trả lời</div><div class="answer-main">'
        + html.escape(str(result.get("answer") or "Không có câu trả lời."))
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="answer-meta">'
        + f'<span class="meta-pill">Intent · {html.escape(str(result.get("intent", "unknown")))}</span>'
        + f'<span class="meta-pill">Độ tin cậy · {float(result.get("confidence", 0.0)):.2f}</span>'
        + "</div>",
        unsafe_allow_html=True,
    )
    direct_audio_paths = [
        Path(str(value)) for value in result.get("direct_evidence_audio_paths", [])
        if str(value).strip() and Path(str(value)).is_file()
    ]

    if direct_audio_paths:
        st.markdown('<div class="evidence-label">Giọng nói đã tách</div>', unsafe_allow_html=True)
        for index, direct_path in enumerate(direct_audio_paths, 1):
            st.audio(str(direct_path), format="audio/wav")
            st.caption(f"Stem người nói {index} · nghe lại để kiểm chứng transcript")
            render_audio_download(
                direct_path,
                f"Tải audio người nói {index}",
                f"audio_chat_download_speaker_{index}_{direct_path.name}",
            )
    spans = list(result.get("evidence", []))
    if spans and not direct_audio_paths:
        st.markdown('<div class="evidence-label">Evidence nghe lại</div>', unsafe_allow_html=True)
        target_labels = tuple(
            str(label) for label in result.get("evidence_labels", [])
            if str(label) in COMBINED_ONTOLOGY
        )
        separator_labels = tuple(
            EVIDENCE_LABEL_ALIASES.get(label, label) for label in target_labels
        )
        method = str(st.session_state.audio_chat_evidence_method)
        for index, span in enumerate(spans, 1):
            start, end = float(span[0]), float(span[1])
            if not target_labels:
                st.warning("Evidence này chưa có nhãn semantic đủ tin cậy để chạy bộ tách.")
                continue
            if not evidence_service_ok:
                st.error("Dịch vụ tách evidence đang khởi động; audio gốc không được dùng để giả làm kết quả.")
                continue
            try:
                with st.spinner(f"Đang tạo evidence bằng {EVIDENCE_METHOD_NAMES[method]}…"):
                    separated = post_evidence_separation(
                        str(audio_path), start, end, separator_labels, method,
                    )
                output_path = Path(str(separated["output_path"]))
                st.audio(str(output_path), format="audio/wav")
                render_audio_download(
                    output_path,
                    f"Tải audio evidence {index}",
                    f"audio_chat_download_evidence_{index}_{output_path.name}",
                )
                readable_labels = ", ".join(
                    CLASS_NAMES_VI.get(label, label.replace("_", " "))
                    for label in separated.get("labels", target_labels)
                )
                st.markdown(
                    '<div class="evidence-caption">'
                    f'Đoạn {index} · {start:.2f}–{end:.2f} giây · '
                    f'Mục tiêu tách: {html.escape(readable_labels)} · '
                    f'{html.escape(EVIDENCE_METHOD_NAMES[method])}</div>',
                    unsafe_allow_html=True,
                )
            except (KeyError, OSError, RuntimeError, urllib.error.URLError):
                st.warning("Không tạo được audio evidence bằng phương pháp đã chọn.")


def render_speaker_separation(
    audio_path: Path,
    prediction: dict[str, Any],
    multispeaker_ok: bool,
    overlap_ok: bool,
) -> None:
    """Show multi-speaker region stems, with the old two-source path as fallback."""

    has_speech = bool(prediction.get("speech_span_seconds"))
    if not has_speech:
        return
    st.markdown("#### Các giọng nói được phân tách")
    result = st.session_state.get("audio_chat_multispeaker_result")
    method = "region"
    if not result:
        result = st.session_state.get("audio_chat_overlap_result")
        method = "two_source"
    if not result:
        separation_error = st.session_state.get("audio_chat_multispeaker_error")
        if separation_error:
            st.warning("Tự động phân cụm giọng chưa hoàn tất: " + str(separation_error))
        st.caption(
            "Hệ thống đã tự động thử phân cụm giọng ngay sau khi tải audio. "
            "Bạn có thể chạy lại thủ công nếu lần tự động gặp lỗi."
        )
        if not multispeaker_ok:
            st.warning("Dịch vụ phân cụm nhiều giọng đang khởi động. Vui lòng thử lại sau ít phút.")
            return
        if st.button(
            "Phân cụm và nghe riêng từng giọng",
            type="primary",
            use_container_width=True,
            key="audio_chat_multispeaker_separate",
        ):
            analysis_progress = st.progress(10, text="Đang phân tích audio…")
            try:
                analysis_progress.progress(45, text="Đang phân tích audio…")
                result = post_multispeaker_region_separation(
                    str(audio_path),
                    json.dumps(prediction, ensure_ascii=False, sort_keys=True),
                )
                analysis_progress.progress(95, text="Đang phân tích audio…")
                prediction = attach_region_speakers(prediction, result)
                st.session_state.audio_chat_prediction = prediction
                st.session_state.audio_chat_multispeaker_result = result
                st.session_state.audio_chat_multispeaker_error = None
                st.session_state.audio_chat_messages = []
                analysis_progress.progress(100, text="Đang phân tích audio…")
                analysis_progress.empty()
                st.rerun()
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                analysis_progress.empty()
                st.error(f"Không phân cụm được giọng nói: {error}")
        return

    speakers = list(result.get("speakers", []))
    if method == "region":
        count = int(result.get("speaker_count") or len(speakers))
        st.caption(
            f"Hệ thống ước lượng {count} cụm giọng. Đây là số cụm âm sắc, không phải "
            "ground truth số người; một người đổi giọng có thể bị chia thành hai cụm."
        )
    elif st.session_state.get("audio_chat_multispeaker_error"):
        st.caption(str(st.session_state.audio_chat_multispeaker_error))
    for row_start in range(0, len(speakers), 2):
        columns = st.columns(2)
        for column, row in zip(columns, speakers[row_start : row_start + 2]):
            with column:
                st.markdown(f"**{str(row.get('display_name') or 'Người nói')}**")
                turns = list(row.get("turns", []))
                if turns:
                    ranges = ", ".join(
                        f"{float(turn['start_seconds']):.2f}–{float(turn['end_seconds']):.2f}s"
                        for turn in turns
                    )
                    st.caption("Vùng giọng: " + ranges)
                path = Path(str(row.get("audio_path") or ""))
                if path.is_file():
                    st.audio(str(path), format="audio/wav")
                    render_audio_download(
                        path,
                        "Tải audio giọng này",
                        f"audio_chat_download_cluster_{row_start}_{path.name}",
                    )
                transcript = str(row.get("hypothesis") or "").strip()
                st.caption("Bản nháp nhận dạng lời nói")
                st.markdown(transcript or "*Chưa nhận dạng được nội dung.*")
    warning = str(result.get("warning") or "Transcript có thể sai khi giọng bị che mạnh.")
    st.info(warning)


def _speaker_span(row: Mapping[str, Any], duration: float) -> tuple[float, float]:
    turns = list(row.get("turns", []))
    if not turns:
        return 0.0, duration
    return (
        min(float(turn.get("start_seconds", 0.0)) for turn in turns),
        max(float(turn.get("end_seconds", duration)) for turn in turns),
    )


def attach_region_speakers(
    prediction: dict[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach estimated region clusters and replace stale one/two-speaker turns."""

    duration = float(result.get("duration_seconds") or 0.0)
    separated: list[dict[str, Any]] = []
    for row in result.get("speakers", []):
        start, end = _speaker_span(row, duration)
        separated.append(
            {
                **row,
                "start_seconds": start,
                "end_seconds": end,
                "words": [],
                "predicted_speaker": "unknown",
            }
        )
    prediction["separated_speakers"] = separated
    prediction["speaker_count"] = int(result.get("speaker_count") or len(separated))
    prediction["speaker_count_is_estimated"] = True
    turns = list(result.get("diarization", {}).get("turns", []))
    prediction["speaker_turns"] = turns
    prediction["predicted_events"] = [
        row for row in prediction.get("predicted_events", [])
        if str(row.get("label") or "") != "Speech"
    ]
    for turn in turns:
        speaker_id = str(turn.get("speaker_id") or "SPEAKER_01")
        prediction["predicted_events"].append(
            {
                "label": "Speech",
                "display_label": "Tiếng người nói · " + speaker_display(speaker_id),
                "start_seconds": float(turn.get("start_seconds", 0.0)),
                "end_seconds": float(turn.get("end_seconds", 0.0)),
                "speaker_id": speaker_id,
                "predicted_speaker": "unknown",
                "source": "multispeaker_region_clustering_v1",
            }
        )
    prediction["predicted_events"].sort(
        key=lambda row: (
            float(row.get("start_seconds", 0.0)),
            float(row.get("end_seconds", 0.0)),
            str(row.get("label") or ""),
        )
    )
    return prediction


def initialize_state() -> None:
    defaults = {
        "audio_chat_messages": [],
        "audio_chat_prediction": None,
        "audio_chat_path": None,
        "audio_chat_digest": None,
        "audio_chat_source": None,
        "audio_chat_duration": None,
        "audio_chat_overlap_result": None,
        "audio_chat_overlap_error": None,
        "audio_chat_multispeaker_result": None,
        "audio_chat_multispeaker_error": None,
        "audio_chat_theme": "Sáng",
        "audio_chat_evidence_method": "ours",
        "audio_chat_question": "",
        "audio_chat_submit": None,
        "audio_chat_source_widget_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation(remove_audio: bool = False) -> None:
    st.session_state.audio_chat_messages = []
    if remove_audio:
        st.session_state.audio_chat_prediction = None
        st.session_state.audio_chat_path = None
        st.session_state.audio_chat_digest = None
        st.session_state.audio_chat_source = None
        st.session_state.audio_chat_duration = None
        st.session_state.audio_chat_overlap_result = None
        st.session_state.audio_chat_overlap_error = None
        st.session_state.audio_chat_multispeaker_result = None
        st.session_state.audio_chat_multispeaker_error = None


def attach_separated_speakers(
    prediction: dict[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach real separated stems to the live prediction contract."""

    duration = float(result.get("duration_seconds") or 0.0)
    prediction["separated_speakers"] = [
        {
            **row,
            "start_seconds": 0.0,
            "end_seconds": duration,
            "words": [],
            "predicted_speaker": "unknown",
        }
        for row in result.get("speakers", [])
    ]
    return prediction


def analyze_audio(raw: bytes, suffix: str, source_name: str) -> None:
    audio_path, run_dir, digest = materialize_live_audio(raw, suffix)
    current_prediction = st.session_state.get("audio_chat_prediction") or {}
    automatic_separation_complete = (
        not bool(current_prediction.get("speech_span_seconds"))
        or bool(st.session_state.get("audio_chat_multispeaker_result"))
    )
    if (
        digest == st.session_state.audio_chat_digest
        and current_prediction.get("format") == "qces_live_annotation_free_prediction_v3"
        and automatic_separation_complete
    ):
        return
    analysis_progress = st.progress(8, text="Đang phân tích audio…")
    try:
        prediction = post_live_inference(audio_path)
    except Exception:
        analysis_progress.empty()
        raise
    analysis_progress.progress(38, text="Đang phân tích audio…")
    baseline_prediction = json.loads(json.dumps(prediction, ensure_ascii=False))
    overlap_result: dict[str, Any] | None = None
    overlap_error: str | None = None
    multispeaker_result: dict[str, Any] | None = None
    multispeaker_error: str | None = None
    duration = audio_duration(audio_path)
    verified_two_speakers = bool(
        prediction.get("diarization", {})
        .get("speaker_verification", {})
        .get("two_speakers_accepted")
    )
    if prediction.get("speech_span_seconds") and duration <= 10.1:
        if multispeaker_region_health():
            try:
                analysis_progress.progress(55, text="Đang phân tích audio…")
                multispeaker_result = post_multispeaker_region_separation(
                    str(audio_path),
                    json.dumps(prediction, ensure_ascii=False, sort_keys=True),
                )
                analysis_progress.progress(84, text="Đang phân tích audio…")
                prediction = attach_region_speakers(prediction, multispeaker_result)
                if (
                    int(multispeaker_result.get("speaker_count") or 0) <= 2
                    and verified_two_speakers
                ):
                    # For an independently verified two-source scene, the
                    # full-recording separator preserves a short overlapped
                    # voice better than hard region masks. Region clustering
                    # wins only when it provides evidence for >2 identities.
                    multispeaker_error = (
                        "Hệ thống xác nhận đúng hai nguồn và dùng bộ tách toàn đoạn để "
                        "giữ giọng ngắn/chồng lấn rõ hơn; phân cụm theo vùng chỉ được "
                        "ưu tiên khi phát hiện hơn hai cụm."
                    )
                    multispeaker_result = None
                    prediction = baseline_prediction
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                multispeaker_error = f"{type(error).__name__}: {error}"
        else:
            multispeaker_error = "Dịch vụ phân cụm nhiều giọng chưa sẵn sàng."
    elif prediction.get("speech_span_seconds"):
        multispeaker_error = (
            "Bộ phân cụm nhiều giọng hiện hỗ trợ audio tối đa 10 giây; "
            f"file này dài {duration:.1f} giây."
        )
    if multispeaker_result is None and int(prediction.get("speaker_count") or 0) >= 2:
        if duration <= 10.1 and overlap_separator_health():
            try:
                analysis_progress.progress(88, text="Đang phân tích audio…")
                overlap_result = post_overlap_separation(str(audio_path))
                prediction = attach_separated_speakers(prediction, overlap_result)
                analysis_progress.progress(96, text="Đang phân tích audio…")
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                overlap_error = f"{type(error).__name__}: {error}"
    (run_dir / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reset_conversation(remove_audio=False)
    st.session_state.audio_chat_prediction = prediction
    st.session_state.audio_chat_path = str(audio_path)
    st.session_state.audio_chat_digest = digest
    st.session_state.audio_chat_source = source_name
    st.session_state.audio_chat_duration = audio_duration(audio_path)
    st.session_state.audio_chat_overlap_result = overlap_result
    st.session_state.audio_chat_overlap_error = overlap_error
    st.session_state.audio_chat_multispeaker_result = multispeaker_result
    st.session_state.audio_chat_multispeaker_error = multispeaker_error
    st.session_state.audio_chat_source_widget_version += 1
    analysis_progress.progress(100, text="Đang phân tích audio…")
    analysis_progress.empty()


def render_sidebar(
    question_service_ok: bool,
    inference_ok: bool,
    evidence_ok: bool,
    overlap_ok: bool,
    multispeaker_ok: bool,
) -> None:
    with st.sidebar:
        render_brand()
        st.markdown('<div class="sidebar-label">Giao diện</div>', unsafe_allow_html=True)
        st.radio(
            "Chế độ màu",
            ("Sáng", "Tối"),
            horizontal=True,
            key="audio_chat_theme",
            label_visibility="collapsed",
        )
        if st.session_state.audio_chat_prediction is not None:
            st.markdown('<div class="sidebar-label">Phiên hiện tại</div>', unsafe_allow_html=True)
            st.caption(str(st.session_state.audio_chat_source or "Audio đã phân tích"))
            if st.button("Xoá hội thoại", use_container_width=True):
                reset_conversation(remove_audio=False)
                st.rerun()
            if st.button("Gỡ audio", use_container_width=True):
                reset_conversation(remove_audio=True)
                st.rerun()
        render_supported_classes()
        render_service_rows(
            question_service_ok, inference_ok, evidence_ok, overlap_ok, multispeaker_ok
        )


def main() -> None:
    st.set_page_config(
        page_title="QCES · AudioQA",
        page_icon=brand_favicon(),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    inject_styles(str(st.session_state.audio_chat_theme))
    question_service_ok = qwen_health()
    inference_ok = live_inference_health()
    evidence_ok = evidence_separator_health()
    overlap_ok = overlap_separator_health()
    multispeaker_ok = multispeaker_region_health()
    render_sidebar(
        question_service_ok, inference_ok, evidence_ok, overlap_ok, multispeaker_ok
    )

    badge_class = "live-badge" if question_service_ok and inference_ok else "live-badge offline-badge"
    badge_text = "Hệ thống sẵn sàng" if question_service_ok and inference_ok else "Hệ thống đang khởi động"
    st.markdown(
        f"""
        <div class="topbar">
          <div><h1>Trò chuyện với âm thanh</h1><p>Hỏi về lời nói, sự kiện và quan hệ thời gian trong cùng một đoạn audio.</p></div>
          <div class="{badge_class}"><span class="live-dot"></span>{badge_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not question_service_ok or not inference_ok:
        st.error("Một dịch vụ xử lý chưa sẵn sàng. Vui lòng thử lại sau ít phút.")
        st.stop()

    prediction = st.session_state.audio_chat_prediction
    audio_path_text = st.session_state.audio_chat_path
    if prediction is None or audio_path_text is None:
        st.markdown(
            """
            <section class="hero-shell">
              <div class="eyebrow">Evidence-grounded AudioQA</div>
              <h2>Không chỉ trả lời. Hệ thống cho bạn nghe phần audio làm bằng chứng.</h2>
              <p>Thu một tình huống thực tế hoặc tải file có tiếng người và âm thanh môi trường. Hệ thống phát hiện timeline, hiểu câu hỏi tự nhiên và trả về đoạn evidence có thể kiểm chứng.</p>
              <div class="cap-row">
                <span class="cap-pill">Hiểu tiếng Việt & tiếng Anh</span>
                <span class="cap-pill">Nhận dạng lời nói</span>
                <span class="cap-pill">Grounding theo thời gian</span>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.info("Thu âm hoặc chọn tệp ngay trên thanh bên dưới. Audio sẽ được phân tích tự động.")
        raw, suffix, source_name = render_composer(False, inference_ok)
        if raw is not None:
            try:
                analyze_audio(raw, suffix, source_name)
                st.rerun()
            except (OSError, RuntimeError, subprocess.CalledProcessError, urllib.error.URLError) as error:
                st.error(f"Không phân tích được audio: {error}")
        return

    audio_path = Path(str(audio_path_text))
    duration = float(st.session_state.audio_chat_duration or audio_duration(audio_path))
    render_timeline(prediction, duration)
    st.audio(str(audio_path), format="audio/wav")
    st.caption("Audio gốc của phiên hiện tại · phát lại trước khi đặt câu hỏi nếu cần")
    render_speaker_separation(audio_path, prediction, multispeaker_ok, overlap_ok)
    st.markdown(
        '<div class="evidence-method-card">'
        '<div class="evidence-method-title">Chọn cách tạo audio evidence</div>'
        '<div class="evidence-method-help">Hai phương pháp dùng cùng nhãn sự kiện và cùng khoảng thời gian do hệ thống dự đoán.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.radio(
        "Phương pháp tách evidence",
        tuple(EVIDENCE_METHOD_NAMES),
        format_func=lambda method: EVIDENCE_METHOD_NAMES[str(method)],
        horizontal=True,
        key="audio_chat_evidence_method",
        label_visibility="collapsed",
    )

    messages = list(st.session_state.audio_chat_messages)
    if not messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Mình đã nghe xong audio. Bạn có thể hỏi về **các âm thanh xuất hiện**, "
                "**sự kiện trước/sau**, **nội dung lời nói** hoặc **giới tính giọng nói**."
            )

        suggestions = [
            "Có những âm thanh gì?",
            "Người nói đã nói gì?",
            "Có bao nhiêu người nói?",
        ]
        columns = st.columns(3)
        for column, suggestion in zip(columns, suggestions):
            if column.button(suggestion, use_container_width=True, key="suggest_" + hashlib.md5(suggestion.encode()).hexdigest()):
                st.session_state.audio_chat_pending = suggestion
                st.rerun()

    for message in messages:
        with st.chat_message(str(message["role"])):
            if message["role"] == "assistant":
                render_assistant_result(message["result"], audio_path, evidence_ok)
            else:
                st.markdown(str(message["content"]))

    raw, suffix, source_name = render_composer(True, inference_ok)
    if raw is not None:
        previous_digest = st.session_state.audio_chat_digest
        try:
            analyze_audio(raw, suffix, source_name)
            if st.session_state.audio_chat_digest != previous_digest:
                st.rerun()
        except (OSError, RuntimeError, subprocess.CalledProcessError, urllib.error.URLError) as error:
            st.error(f"Không phân tích được audio: {error}")

    pending = st.session_state.pop("audio_chat_pending", None)
    submitted = st.session_state.pop("audio_chat_submit", None)
    question = str(pending or submitted or "").strip()
    if question:
        st.session_state.audio_chat_messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Đang hiểu câu hỏi và tìm evidence…"):
                result = answer_question(question, audio_path, prediction)
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            result = {
                "answer": "Mình chưa thể xử lý câu hỏi lúc này.",
                "intent": "error",
                "confidence": 0.0,
                "evidence": [],
                "note": f"Backend error: {error}",
            }
        st.session_state.audio_chat_messages.append({"role": "assistant", "result": result})
        st.rerun()


if __name__ == "__main__":
    main()
