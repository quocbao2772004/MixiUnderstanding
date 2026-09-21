#!/usr/bin/env python3
"""One-page VieNeu automotive stress-test and AudioSep result browser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / "data/qces_vietnamese_vieneu_source_v3"
DATASET_DIR = PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_stress_v2"
AUDIT_DIR = PROJECT_ROOT / "outputs/qces_vietnamese_vieneu_source_v3/audit"
RESULT_DIR = PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_audiosep_v1"
PROFILES = {
    "Thực tế hơn · speech +6 dB · SNR trung bình −4 dB": {
        "dataset": PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_realistic_v1",
        "result": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_realistic_audiosep_v1",
        "audiosep_oa": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_audiosep_oa_v1",
        "frcrn": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_frcrn_zero_shot_v1",
        "frcrn_oa": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_frcrn_oa_v1",
        "description": "Lời nói gần microphone hơn; noise, còi và bố cục giữ nguyên bản stress.",
    },
    "Stress cũ · SNR trung bình −10 dB": {
        "dataset": DATASET_DIR,
        "result": RESULT_DIR,
        "audiosep_oa": None,
        "frcrn": None,
        "frcrn_oa": None,
        "description": "Ca cực khó: lời nói nhỏ hơn đáng kể so với tổng nhiễu.",
    },
}

SCENARIO_VI = {
    "navigation": "Chỉ đường",
    "climate": "Điều hòa",
    "dropoff": "Trả khách",
    "traffic": "Tình trạng giao thông",
    "phone_call": "Gọi điện khi lái xe",
    "fuel": "Tìm cây xăng",
    "pickup": "Đón khách",
    "delivery": "Giao hàng",
    "assistant": "Trợ lý trong xe",
    "safety": "Cảnh báo an toàn",
    "weather": "Thời tiết",
    "parking": "Đỗ xe",
}
MODE_NAMES = {
    "mixture_oracle_span": "Mixture + oracle speech span",
    "audiosep_full": "AudioSep evidence (full output)",
    "audiosep_oracle_span": "AudioSep evidence + oracle speech span",
    "clean_speech_upper_bound": "Clean VieNeu upper bound",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@st.cache_data(show_spinner=False)
def load_data(
    dataset_dir_text: str,
    result_dir_text: str,
    audiosep_oa_dir_text: str,
    frcrn_dir_text: str,
    frcrn_oa_dir_text: str,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir_text)
    result_dir = Path(result_dir_text)
    audiosep_oa_dir = Path(audiosep_oa_dir_text) if audiosep_oa_dir_text else None
    frcrn_dir = Path(frcrn_dir_text) if frcrn_dir_text else None
    frcrn_oa_dir = Path(frcrn_oa_dir_text) if frcrn_oa_dir_text else None
    source_receipt = _json(SOURCE_DIR / "dataset_receipt.json")
    dataset_receipt = _json(dataset_dir / "dataset_receipt.json")
    audit_receipt = _json(AUDIT_DIR / "audit_receipt.json")
    result_receipt = _json(result_dir / "receipt.json")
    source_scenes = {row["scene_id"]: row for row in _jsonl(SOURCE_DIR / "scenes.jsonl")}
    scenes = _jsonl(dataset_dir / "scenes.jsonl")
    questions: dict[str, list[dict[str, Any]]] = {}
    for row in _jsonl(dataset_dir / "questions.jsonl"):
        questions.setdefault(row["scene_id"], []).append(row)
    asr: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _jsonl(result_dir / "asr_items.jsonl"):
        asr.setdefault(row["scene_id"], {})[row["mode"]] = row
    separator = {
        row["scene_id"]: row
        for row in _jsonl(result_dir / "separator_items.jsonl")
        if row["prompt"] == result_receipt["selected_prompt"]
    }
    audiosep_oa_receipt = _json(audiosep_oa_dir / "receipt.json") if audiosep_oa_dir else None
    frcrn_receipt = _json(frcrn_dir / "receipt.json") if frcrn_dir else None
    frcrn_oa_receipt = _json(frcrn_oa_dir / "receipt.json") if frcrn_oa_dir else None
    audiosep_oa_asr: dict[str, dict[str, dict[str, Any]]] = {}
    if audiosep_oa_dir:
        for row in _jsonl(audiosep_oa_dir / "asr_items.jsonl"):
            audiosep_oa_asr.setdefault(row["scene_id"], {})[row["mode"]] = row
    frcrn_oa_asr: dict[str, dict[str, dict[str, Any]]] = {}
    if frcrn_oa_dir:
        for row in _jsonl(frcrn_oa_dir / "asr_items.jsonl"):
            frcrn_oa_asr.setdefault(row["scene_id"], {})[row["mode"]] = row
    return {
        "source_receipt": source_receipt,
        "dataset_receipt": dataset_receipt,
        "audit_receipt": audit_receipt,
        "result_receipt": result_receipt,
        "source_scenes": source_scenes,
        "scenes": scenes,
        "questions": questions,
        "asr": asr,
        "separator": separator,
        "result_dir": result_dir,
        "audiosep_oa_dir": audiosep_oa_dir,
        "audiosep_oa_receipt": audiosep_oa_receipt,
        "audiosep_oa_asr": audiosep_oa_asr,
        "frcrn_dir": frcrn_dir,
        "frcrn_receipt": frcrn_receipt,
        "frcrn_oa_dir": frcrn_oa_dir,
        "frcrn_oa_receipt": frcrn_oa_receipt,
        "frcrn_oa_asr": frcrn_oa_asr,
    }


if not globals().get("EMBEDDED_IN_QCES_DEMO", False):
    st.set_page_config(page_title="VieNeu Automotive Evidence Demo", page_icon="🚗", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1450px; padding-top: 1.3rem; padding-bottom: 3rem;}
    .flow {padding:.8rem 1rem; border:1px solid #bfd4ef; border-radius:.65rem;
           background:#eef6ff; color:#17385f; margin-bottom:1rem;}
    .target {padding:.8rem 1rem; border-left:5px solid #3379c5; background:#f5f8fc;
             border-radius:.3rem; font-size:1.08rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

profile_name = st.radio("Mức nhiễu", list(PROFILES), horizontal=True)
profile = PROFILES[profile_name]
data = load_data(
    str(profile["dataset"]),
    str(profile["result"]),
    str(profile["audiosep_oa"]) if profile["audiosep_oa"] else "",
    str(profile["frcrn"]) if profile["frcrn"] else "",
    str(profile["frcrn_oa"]) if profile["frcrn_oa"] else "",
)
source_receipt = data["source_receipt"]
dataset_receipt = data["dataset_receipt"]
audit_receipt = data["audit_receipt"]
result_receipt = data["result_receipt"]

st.title("🚗 VieNeu: hiểu lời nói tiếng Việt trong xe có nhiễu")
st.markdown(
    '<div class="flow"><b>Câu nói VieNeu</b> → traffic + động cơ + còi + siren → '
    '<b>AudioSep hoặc FRCRN</b> → thêm lại 25% mixture (OA) → Whisper-medium</div>',
    unsafe_allow_html=True,
)
st.warning(
    "Đây là controlled synthetic stress test để phát triển model. Kết quả paper/product vẫn phải "
    "được xác nhận trên speech thu thật trong cabin; clean TTS không được coi là real-speech test set."
)
st.info(f"**Thiết lập đang nghe:** {profile_name}. {profile['description']}")

test_sep = result_receipt["test_separator"]
test_asr = result_receipt["asr"]["test"]
metrics = st.columns(7)
metrics[0].metric("Scenes", dataset_receipt["scene_count"])
metrics[1].metric("Tình huống", len(source_receipt["scenario_categories"]))
metrics[2].metric("Giọng VieNeu", len(source_receipt["voice_counts"]))
metrics[3].metric("Clean audit đạt ↑", f'{100*audit_receipt["acceptance_rate_↑"]:.0f}%')
metrics[4].metric("Clean WER ↓", f'{audit_receipt["mean_wer_↓"]:.3f}')
metrics[5].metric("AudioSep SI-SDRi ↑", f'{test_sep["mean_si_sdri_db_↑"]:+.2f} dB')
metrics[6].metric("SNR nhỏ nhất", f'{dataset_receipt["min_measured_speech_to_all_interference_snr_db"]:.1f} dB')

summary_rows = [
        {
            "Input cho Whisper-medium": "Mixture + oracle speech span",
            "WER ↓": test_asr["mixture_oracle_span"]["mean_wer_↓"],
            "Accuracy@WER≤0.25 ↑": test_asr["mixture_oracle_span"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "Baseline nghe audio ồn",
        },
        {
            "Input cho Whisper-medium": "AudioSep evidence (ours pipeline)",
            "WER ↓": test_asr["audiosep_full"]["mean_wer_↓"],
            "Accuracy@WER≤0.25 ↑": test_asr["audiosep_full"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "Frozen AudioSep, prompt: a person speaking",
        },
        {
            "Input cho Whisper-medium": "AudioSep evidence + oracle speech span",
            "WER ↓": test_asr["audiosep_oracle_span"]["mean_wer_↓"],
            "Accuracy@WER≤0.25 ↑": test_asr["audiosep_oracle_span"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "So sánh công bằng cùng đúng vùng lời nói",
        },
        {
            "Input cho Whisper-medium": "Clean VieNeu upper bound",
            "WER ↓": test_asr["clean_speech_upper_bound"]["mean_wer_↓"],
            "Accuracy@WER≤0.25 ↑": test_asr["clean_speech_upper_bound"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "Trần ASR khi không có nhiễu",
        },
]
if data["audiosep_oa_receipt"] and data["frcrn_oa_receipt"]:
    aoa = data["audiosep_oa_receipt"]
    foa = data["frcrn_oa_receipt"]
    summary_rows = [
        {
            "Input cho Whisper-medium": "Mixture + cùng speech span",
            "Corpus WER ↓": aoa["summary"]["test"]["mixture_beta_1"]["corpus_wer_↓"],
            "Accuracy@WER≤0.25 ↑": aoa["summary"]["test"]["mixture_beta_1"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "Không enhancement",
        },
        {
            "Input cho Whisper-medium": "AudioSep thuần",
            "Corpus WER ↓": aoa["summary"]["test"]["audiosep_beta_0"]["corpus_wer_↓"],
            "Accuracy@WER≤0.25 ↑": aoa["summary"]["test"]["audiosep_beta_0"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "Separator frozen",
        },
        {
            "Input cho Whisper-medium": "AudioSep + OA β=0.25",
            "Corpus WER ↓": aoa["summary"]["test"]["audiosep_oa_beta_0_25"]["corpus_wer_↓"],
            "Accuracy@WER≤0.25 ↑": aoa["summary"]["test"]["audiosep_oa_beta_0_25"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "75% enhanced + 25% mixture",
        },
        {
            "Input cho Whisper-medium": "FRCRN thuần",
            "Corpus WER ↓": foa["summary"]["test"]["frcrn_beta_0"]["corpus_wer_↓"],
            "Accuracy@WER≤0.25 ↑": foa["summary"]["test"]["frcrn_beta_0"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "Speech enhancer 16 kHz frozen",
        },
        {
            "Input cho Whisper-medium": "FRCRN + OA β=0.25",
            "Corpus WER ↓": foa["summary"]["test"]["frcrn_oa_beta_0_25"]["corpus_wer_↓"],
            "Accuracy@WER≤0.25 ↑": foa["summary"]["test"]["frcrn_oa_beta_0_25"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "Best fixed pipeline hiện tại",
        },
        {
            "Input cho Whisper-medium": "Clean upper bound",
            "Corpus WER ↓": foa["summary"]["test"]["clean_upper_bound"]["corpus_wer_↓"],
            "Accuracy@WER≤0.25 ↑": foa["summary"]["test"]["clean_upper_bound"]["accuracy_at_wer_0.25_↑"],
            "Vai trò": "Không có nhiễu",
        },
    ]
summary = pd.DataFrame(summary_rows)
with st.expander("Kết quả tổng trên 12 test scenes", expanded=True):
    st.caption(
        "WER càng thấp càng tốt. OA (observation adding) trộn lại 25% mixture đã căn RMS để giảm "
        "artifact và phục hồi chi tiết âm vị mà enhancer có thể làm mất."
    )
    summary_formats = {"Accuracy@WER≤0.25 ↑": "{:.1%}"}
    if "WER ↓" in summary.columns:
        summary_formats["WER ↓"] = "{:.3f}"
    if "Corpus WER ↓" in summary.columns:
        summary_formats["Corpus WER ↓"] = "{:.3f}"
    st.dataframe(
        summary.style.format(summary_formats),
        hide_index=True,
        use_container_width=True,
    )

split = st.radio("Split", ["test", "val"], horizontal=True)
scenes = [scene for scene in data["scenes"] if scene["split"] == split]
scene_lookup = {scene["scene_id"]: scene for scene in scenes}


def _scene_label(scene_id: str) -> str:
    scene = scene_lookup[scene_id]
    source = data["source_scenes"][scene["paired_scene_id"]]
    speech = source["events"][0]
    category = SCENARIO_VI.get(source["scenario_category"], source["scenario_category"])
    return f"{scene_id} · {scene['difficulty'].replace('automotive_', '')} · {category} · {speech['tts_voice']}"


scene_ids = list(scene_lookup)
default_index = min(2, len(scene_ids) - 1)
if data["frcrn_oa_asr"]:
    improvements: list[tuple[float, str]] = []
    for candidate_scene_id in scene_ids:
        candidate_rows = data["frcrn_oa_asr"].get(candidate_scene_id, {})
        mixture_row = candidate_rows.get("mixture_beta_1")
        enhanced_row = candidate_rows.get("frcrn_oa_beta_0_25")
        if mixture_row and enhanced_row:
            improvements.append((mixture_row["wer_↓"] - enhanced_row["wer_↓"], candidate_scene_id))
    if improvements:
        default_scene_id = max(improvements)[1]
        default_index = scene_ids.index(default_scene_id)
scene_id = st.selectbox("Chọn case", scene_ids, format_func=_scene_label, index=default_index)
scene = scene_lookup[scene_id]
source_scene = data["source_scenes"][scene["paired_scene_id"]]
speech_source = source_scene["events"][0]
transcript = speech_source["transcript"]

st.subheader("1. Bối cảnh và yêu cầu")
info = st.columns(4)
info[0].metric("Tình huống", SCENARIO_VI.get(source_scene["scenario_category"], source_scene["scenario_category"]))
info[1].metric("Giọng", f'{speech_source["tts_voice"]} ({speech_source["speaker_group"]})')
info[2].metric("Difficulty", scene["difficulty"].replace("automotive_", ""))
info[3].metric("Measured SNR", f'{scene["measured_speech_to_all_interference_snr_db"]:.2f} dB')
st.markdown(f'<div class="target"><b>Câu hỏi:</b> Người trong xe đã nói gì?<br><b>Đáp án chuẩn:</b> {transcript}</div>', unsafe_allow_html=True)

st.subheader("2. Nghe trực tiếp các đầu vào ASR")
clean_path = _resolve(speech_source["stem_path"])
mixture_path = _resolve(scene["mixture_path"])
enhanced_path = data["result_dir"] / "audio/selected" / split / f"{scene_id}.flac"
players_top = st.columns(3)
with players_top[0]:
    st.markdown("**A. Clean VieNeu — upper bound**")
    st.caption("Không có nhiễu; dùng để xác nhận TTS nói đúng câu.")
    st.audio(str(clean_path))
with players_top[1]:
    st.markdown("**B. Mixture gốc**")
    st.caption("Speech cùng traffic, engine, horn và siren.")
    st.audio(str(mixture_path))
with players_top[2]:
    st.markdown("**C. AudioSep thuần**")
    st.caption("Frozen separator, text prompt: `a person speaking`.")
    st.audio(str(enhanced_path))

has_enhancement_comparison = bool(data["audiosep_oa_dir"] and data["frcrn_dir"] and data["frcrn_oa_dir"])
if has_enhancement_comparison:
    audiosep_oa_path = (
        data["audiosep_oa_dir"] / "audio/audiosep_oa_beta_0_25" / split / f"{scene_id}.flac"
    )
    frcrn_path = data["frcrn_dir"] / "audio/selected" / split / f"{scene_id}.flac"
    frcrn_oa_path = data["frcrn_oa_dir"] / "audio/frcrn_oa_beta_0_25" / split / f"{scene_id}.flac"
    players_bottom = st.columns(3)
    with players_bottom[0]:
        st.markdown("**D. AudioSep + OA β=0.25**")
        st.caption("75% AudioSep + 25% mixture đã căn RMS.")
        st.audio(str(audiosep_oa_path))
    with players_bottom[1]:
        st.markdown("**E. FRCRN thuần**")
        st.caption("Speech enhancement chuyên dụng 16 kHz, không dùng transcript.")
        st.audio(str(frcrn_path))
    with players_bottom[2]:
        st.markdown("**F. FRCRN + OA β=0.25 — tốt nhất hiện tại**")
        st.caption("75% FRCRN + 25% mixture đã căn RMS; fixed setting từ validation.")
        st.audio(str(frcrn_oa_path))

separator = data["separator"].get(scene_id)
if separator:
    cols = st.columns(3)
    cols[0].metric("Case SI-SDRi ↑", f'{separator["si_sdri_db_↑"]:+.2f} dB')
    cols[1].metric("Case SD-SDRi ↑", f'{separator["sd_sdri_db_↑"]:+.2f} dB')
    cols[2].metric("Output/target energy ↔", f'{separator["output_to_target_energy_ratio_db_↔"]:+.2f} dB')

st.subheader("3. Whisper-medium nghe từng input")
asr_rows = []
if has_enhancement_comparison:
    audiosep_rows = data["audiosep_oa_asr"].get(scene_id, {})
    frcrn_rows = data["frcrn_oa_asr"].get(scene_id, {})
    comparison_modes = [
        ("Mixture + cùng speech span", audiosep_rows.get("mixture_beta_1")),
        ("AudioSep thuần", audiosep_rows.get("audiosep_beta_0")),
        ("AudioSep + OA β=0.25", audiosep_rows.get("audiosep_oa_beta_0_25")),
        ("FRCRN thuần", frcrn_rows.get("frcrn_beta_0")),
        ("FRCRN + OA β=0.25", frcrn_rows.get("frcrn_oa_beta_0_25")),
        ("Clean upper bound", frcrn_rows.get("clean_upper_bound")),
    ]
    for label, row in comparison_modes:
        if not row:
            continue
        asr_rows.append(
            {
                "Input": label,
                "Whisper dự đoán": row["hypothesis"],
                "WER ↓": row["wer_↓"],
                "Đạt WER≤0.25": "✅" if row["wer_↓"] <= 0.25 else "❌",
            }
        )
    st.caption(
        "Tất cả hàng dùng cùng speech span và cùng cấu hình Whisper để so sánh công bằng. "
        "Speech span hiện là oracle; khi triển khai thật sẽ thay bằng VAD/predicted span."
    )
else:
    for mode in MODE_NAMES:
        row = data["asr"].get(scene_id, {}).get(mode)
        if not row:
            continue
        asr_rows.append(
            {
                "Input": MODE_NAMES[mode],
                "Whisper dự đoán": row["hypothesis"],
                "WER ↓": row["wer_↓"],
                "Đạt WER≤0.25": "✅" if row["correct_at_wer_0.25"] else "❌",
            }
        )
st.dataframe(pd.DataFrame(asr_rows).style.format({"WER ↓": "{:.3f}"}), hide_index=True, use_container_width=True)

st.subheader("4. Scene gồm những âm thanh gì?")
timeline = []
for event in scene["events"]:
    timeline.append(
        {
            "Vai trò": event["role"],
            "Âm thanh": event["display_name"],
            "Bắt đầu": event["onset_seconds"],
            "Kết thúc": event["offset_seconds"],
            "Chồng lời nói": "✅" if event["role"] in {"continuous_traffic", "continuous_heavy_engine", "truck_horn_overlap", "police_siren_overlap"} else "",
        }
    )
st.dataframe(
    pd.DataFrame(timeline).style.format({"Bắt đầu": "{:.2f}s", "Kết thúc": "{:.2f}s"}),
    hide_index=True,
    use_container_width=True,
)

with st.expander("Các câu hỏi được tạo cho scene này", expanded=False):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Câu hỏi": row["question"],
                    "Loại": row["answer_type"],
                    "Đáp án chuẩn": row["answer"],
                    "No evidence": row["no_evidence"],
                }
                for row in data["questions"].get(scene_id, [])
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "VieNeu v3 Turbo chỉ tạo clean speech. Tất cả tiếng xe là các nguồn AudioSet-derived độc lập; "
    "evidence AudioSep không dùng target transcript hoặc clean stem khi inference."
)
