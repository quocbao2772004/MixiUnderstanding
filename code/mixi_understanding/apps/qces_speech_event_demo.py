#!/usr/bin/env python3
"""Single-page browser for QCES speech + sound evidence results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIENEU_DEMO_NAME = "🆕 VieNeu / tình huống thực tế trong xe"
VARIANTS = {
    "Easy / ít overlap": {
        "dataset": PROJECT_ROOT / "data/qces_speech_event_v1_smoke100",
        "output": PROJECT_ROOT / "outputs/qces_speech_event_v1_pipeline",
    },
    "Hard / noise chồng speech": {
        "dataset": PROJECT_ROOT / "data/qces_speech_event_hard_v2",
        "output": PROJECT_ROOT / "outputs/qces_speech_event_hard_v2_pipeline",
    },
    "Tiếng Việt / còi xe + noise": {
        "dataset": PROJECT_ROOT / "data/qces_vietnamese_speech_event_demo_v1",
        "output": PROJECT_ROOT / "outputs/qces_vietnamese_speech_event_demo_v1_pipeline",
    },
    "Tiếng Việt / automotive cực ồn": {
        "dataset": PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2",
        "output": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_stress_v2_pipeline",
    },
}
AF3_MIXTURE_OUTPUT = (
    PROJECT_ROOT / "outputs/qces_speech_event_hard_v2_af3_mixture"
)
AUTOMOTIVE_ENHANCEMENT_OUTPUTS = {
    "AudioSep": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_audiosep_v1",
    "SepFormer pretrained": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_sepformer_v1",
    "SepFormer domain-adapted": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_sepformer_finetuned_eval_v1",
    "TF magnitude mask": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_tfmask_eval_v1",
    "TF complex mask": PROJECT_ROOT / "outputs/qces_vietnamese_automotive_complexmask_eval_v1",
}
AUTOMOTIVE_ASR_OUTPUTS = {
    "SepFormer domain-adapted": PROJECT_ROOT / "outputs/qces_vietnamese_enhancement_whisper_medium_v1",
    "TF magnitude mask": PROJECT_ROOT / "outputs/qces_vietnamese_tfmask_whisper_medium_v1",
    "TF complex mask": PROJECT_ROOT / "outputs/qces_vietnamese_complexmask_whisper_medium_v1",
}
AUTOMOTIVE_FUSION_OUTPUT = PROJECT_ROOT / "outputs/qces_vietnamese_automotive_fusion_v1"

OPERATION_EXPLANATIONS = {
    "event_before_speech": "Tìm speech span đầu tiên, rồi chọn sound event gần nhất ngay trước nó.",
    "event_after_speech": "Tìm utterance được hỏi, rồi chọn sound event gần nhất sau khi utterance kết thúc.",
    "event_during_speech": "Chọn sound event có overlap lớn nhất với utterance đầu tiên.",
    "event_during_speech_ordinal": "Lấy các sound event bắt đầu trong utterance đầu, sắp theo onset rồi chọn thứ nhất/thứ hai.",
    "event_during_second_speech": "Chọn sound event có tâm thời gian nằm trong utterance thứ hai.",
    "event_between_speech": "Chọn sound event nằm giữa hai speech spans dự đoán.",
    "speech_content_ordinal": "Sắp speech spans theo onset và ASR riêng utterance thứ nhất/thứ hai.",
    "speech_after_quote": "Dùng câu trích dẫn làm anchor và trả transcript của utterance tiếp theo.",
    "speaker_absent": "Scene chỉ có speaker nữ; truy vấn speaker nam phải trả No evidence.",
    "first_event": "Chọn non-speech event có onset sớm nhất.",
    "vi_speech_content": "Tìm speech span, chạy PhoWhisper trên đoạn model dự đoán và trả transcript tiếng Việt.",
    "vi_speech_content_during_horn": "Trả nội dung lời nói trong scene có tiếng còi xe tải chồng trực tiếp lên speech.",
    "vi_speech_content_during_event": "Trả nội dung speech khi sound event được nêu trong câu hỏi đang chồng lên lời nói.",
    "vi_event_before_speech": "Tìm speech span, rồi chọn sound event gần nhất ngay trước lúc người đó bắt đầu nói.",
    "vi_event_after_speech": "Tìm speech span, rồi chọn sound event gần nhất ngay sau lúc người đó nói xong.",
    "vi_second_speaker_absent": "Scene chỉ có một người nói; truy vấn người nói thứ hai phải trả Không có bằng chứng.",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@st.cache_data(show_spinner=False)
def load_data(variant_name: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_dir = Path(VARIANTS[variant_name]["dataset"])
    output_dir = Path(VARIANTS[variant_name]["output"])
    dataset_receipt = json.loads((dataset_dir / "dataset_receipt.json").read_text(encoding="utf-8"))
    pipeline_receipt = json.loads((output_dir / "receipt.json").read_text(encoding="utf-8"))
    scenes = _read_jsonl(dataset_dir / "scenes.jsonl")
    questions = _read_jsonl(dataset_dir / "questions.jsonl")
    results = _read_jsonl(output_dir / "pipeline_results.jsonl")
    return dataset_receipt, pipeline_receipt, scenes, questions, results


@st.cache_data(show_spinner=False)
def load_af3_mixture_results() -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    receipt_path = AF3_MIXTURE_OUTPUT / "receipt.json"
    item_path = AF3_MIXTURE_OUTPUT / "items.jsonl"
    if not receipt_path.is_file() or not item_path.is_file():
        return None, {}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(item_path)
    return receipt, {str(row["question_id"]): row for row in rows}


@st.cache_data(show_spinner=False)
def load_automotive_enhancement() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows = []
    scene_asr: dict[str, dict[str, Any]] = {}
    audiosep = json.loads((AUTOMOTIVE_ENHANCEMENT_OUTPUTS["AudioSep"] / "receipt.json").read_text(encoding="utf-8"))
    rows.append({"Phương pháp": "AudioSep — deploy", "SI-SDRi ↑ (dB)": audiosep["test_separator"]["mean_si_sdri_db_↑"],
                 "WER ↓": 0.7729114729993886, "Chọn": "✅"})
    pretrained = json.loads((AUTOMOTIVE_ENHANCEMENT_OUTPUTS["SepFormer pretrained"] / "receipt.json").read_text(encoding="utf-8"))
    rows.append({"Phương pháp": "SepFormer pretrained", "SI-SDRi ↑ (dB)": pretrained["summary"]["test"]["separator"]["mean_si_sdri_db_↑"],
                 "WER ↓": float("nan"), "Chọn": ""})
    for method, output in AUTOMOTIVE_ENHANCEMENT_OUTPUTS.items():
        if method in ("AudioSep", "SepFormer pretrained"):
            continue
        receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
        separator = receipt["summary"]["test"]
        if "separator" in separator:
            separator = separator["separator"]
        asr_receipt = json.loads((AUTOMOTIVE_ASR_OUTPUTS[method] / "receipt.json").read_text(encoding="utf-8"))
        rows.append({"Phương pháp": method, "SI-SDRi ↑ (dB)": separator["mean_si_sdri_db_↑"],
                     "WER ↓": asr_receipt["test"]["domain_enhanced_full"]["mean_wer_↓"], "Chọn": ""})
        for item in _read_jsonl(AUTOMOTIVE_ASR_OUTPUTS[method] / "items.jsonl"):
            if item["mode"] in ("domain_enhanced_full", "audiosep_full", "clean_upper_bound"):
                scene_asr.setdefault(str(item["scene_id"]), {})[f"{method}:{item['mode']}"] = item
    fusion_items = _read_jsonl(AUTOMOTIVE_FUSION_OUTPUT / "items.jsonl")
    for item in fusion_items:
        if item.get("split") == "test" and item["mode"] == "mixture":
            scene_asr.setdefault(str(item["scene_id"]), {})["mixture"] = item
    return pd.DataFrame(rows), scene_asr


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def short_text(value: str | None, length: int = 86) -> str:
    text = str(value or "")
    return text if len(text) <= length else text[: length - 1] + "…"


st.set_page_config(
    page_title="QCES Speech + Sound Evidence",
    page_icon="🎧",
    layout="wide",
)
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
      .qces-flow {padding: .75rem 1rem; border-radius: .6rem; background: #eef5ff;
                  border: 1px solid #c8dcfa; color: #17345f; margin-bottom: 1rem;}
      .qces-answer {padding: .8rem 1rem; border-radius: .6rem; border: 1px solid #d7dce3;}
      .small-note {font-size: .88rem; color: #586271;}
    </style>
    """,
    unsafe_allow_html=True,
)

variant_name = st.radio(
    "Chọn demo",
    [VIENEU_DEMO_NAME, *VARIANTS],
    horizontal=True,
    help="VieNeu dùng câu nói thực tế trong xe; các lựa chọn còn lại là benchmark speech-event trước đó.",
)
if variant_name == VIENEU_DEMO_NAME:
    import runpy

    runpy.run_path(
        str(PROJECT_ROOT / "code/mixi_understanding/apps/qces_vieneu_automotive_demo.py"),
        init_globals={"EMBEDDED_IN_QCES_DEMO": True},
    )
    st.stop()
dataset_receipt, pipeline_receipt, scenes, questions, results = load_data(variant_name)
af3_receipt, af3_by_question = load_af3_mixture_results()
automotive_benchmark, automotive_asr_by_scene = (load_automotive_enhancement() if "cực ồn" in variant_name else (None, {}))
test_scenes = [scene for scene in scenes if scene["split"] == "test"]
test_results = [row for row in results if row["split"] == "test"]
scene_by_id = {scene["scene_id"]: scene for scene in scenes}
question_by_id = {question["question_id"]: question for question in questions}
results_by_scene: dict[str, list[dict[str, Any]]] = {}
for row in test_results:
    results_by_scene.setdefault(row["scene_id"], []).append(row)
for rows in results_by_scene.values():
    rows.sort(key=lambda row: row["question_id"])

qa = pipeline_receipt["qa"]["test"]
detector = pipeline_receipt["detector"]["test"]

st.title("🎧 QCES Speech + Sound Evidence Demo")
st.markdown(
    '<div class="qces-flow"><b>Mixture audio + question</b> → frozen BEATs-Strong + '
    '11-class temporal head → Speech spans + 10 sound classes → Whisper/PhoWhisper ASR → '
    '<b>answer + predicted temporal evidence</b></div>',
    unsafe_allow_html=True,
)
if variant_name.startswith("Tiếng Việt"):
    if "cực ồn" in variant_name:
        st.warning(
            "Stress test ghép cặp: traffic + heavy engine chạy suốt, truck horn và police siren "
            "chồng speech, SNR còi từ −5 đến −15 dB. Đây là failure-analysis thực tế, không phải cherry-picked demo."
        )
    else:
        st.info(
            "Nhánh tiếng Việt dùng speech thật từ FLEURS vi_vn và PhoWhisper-tiny. "
            "Tiếng còi xe tải được trộn trực tiếp lên lời nói như một stressor ô tô; "
            "head hiện tại chưa có class còi xe, nên demo này không tính còi xe là một event model đã nhận dạng."
        )

if automotive_benchmark is not None:
    with st.expander("Benchmark tách/khử nhiễu trên 12 test scenes", expanded=True):
        st.caption(
            "SI-SDRi càng cao càng sạch về waveform; WER càng thấp càng giữ đúng lời nói. "
            "AudioSep được chọn deploy vì có cân bằng tốt nhất. Các WER có số dùng cùng Whisper-medium; "
            "SepFormer pretrained chưa chạy bằng đúng ASR này nên để trống."
        )
        st.dataframe(
            automotive_benchmark.style.format({"SI-SDRi ↑ (dB)": "{:.2f}", "WER ↓": "{:.3f}"}, na_rep="—"),
            hide_index=True, use_container_width=True,
        )

metric_columns = st.columns(7)
metric_columns[0].metric("QA overall ↑", percent(qa["overall_accuracy_↑"]))
metric_columns[1].metric("Event answer ↑", percent(qa["event_answer_accuracy_↑"]))
metric_columns[2].metric("Speech answer ↑", percent(qa["transcript_accuracy_at_wer_0.25_↑"]))
metric_columns[3].metric("Transcript WER ↓", f"{qa['mean_transcript_wer_↓']:.3f}")
metric_columns[4].metric("No evidence ↑", percent(qa["no_evidence_accuracy_↑"]))
metric_columns[5].metric("Evidence IoU ↑", f"{qa['mean_evidence_iou_↑']:.3f}")
metric_columns[6].metric("Detector event F1 ↑", f"{detector['event_f1_↑']:.3f}")

if variant_name.startswith("Tiếng Việt"):
    comparison_names = ("Tiếng Việt / còi xe + noise", "Tiếng Việt / automotive cực ồn")
    comparison_title = "So sánh ghép cặp: tiếng Việt vừa và automotive cực ồn"
else:
    comparison_names = ("Easy / ít overlap", "Hard / noise chồng speech")
    comparison_title = "So sánh Easy và Hard-overlap"
comparison_rows = []
for name in comparison_names:
    values = load_data(name)[1]["qa"]["test"]
    comparison_rows.append({
        "Test condition": name,
        "QA overall ↑": values["overall_accuracy_↑"],
        "Event answer ↑": values["event_answer_accuracy_↑"],
        "Speech answer ↑": values["transcript_accuracy_at_wer_0.25_↑"],
        "WER ↓": values["mean_transcript_wer_↓"],
        "Evidence IoU ↑": values["mean_evidence_iou_↑"],
    })
comparison = pd.DataFrame(comparison_rows)
with st.expander(comparison_title, expanded=(variant_name.startswith("Hard") or variant_name.startswith("Tiếng Việt"))):
    st.dataframe(
        comparison.style.format(
            {
                "QA overall ↑": "{:.3f}",
                "Event answer ↑": "{:.3f}",
                "Speech answer ↑": "{:.3f}",
                "WER ↓": "{:.3f}",
                "Evidence IoU ↑": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

if variant_name.startswith("Hard") and af3_receipt is not None:
    af3_metrics = af3_receipt["metrics"]
    with st.expander("AF3 nghe trực tiếp mixture hard", expanded=True):
        st.caption(
            "Input của baseline này là Audio Flamingo 3 + toàn bộ mixture gốc + "
            "câu hỏi gốc. AF3 không nhận predicted span/evidence từ hệ của mình. "
            "Chỉ 36 câu hỏi transcript được đánh giá."
        )
        af3_cols = st.columns(4)
        af3_cols[0].metric(
            "AF3 speech acc@WER≤0.25 ↑",
            percent(af3_metrics["utterance_accuracy_at_wer_0.25_↑"]),
        )
        af3_cols[1].metric("AF3 mean WER ↓", f'{af3_metrics["mean_utterance_wer_↓"]:.3f}')
        af3_cols[2].metric("AF3 corpus WER ↓", f'{af3_metrics["corpus_wer_↓"]:.3f}')
        af3_cols[3].metric(
            "AF3 exact transcript ↑",
            percent(af3_metrics["exact_normalized_transcript_accuracy_↑"]),
        )

with st.expander("Dataset và cách đọc metric", expanded=False):
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            - **{dataset_receipt['scene_count']} scenes:** {dataset_receipt['scenes_by_split']}
            - **{dataset_receipt['question_count']} questions**, {dataset_receipt.get('questions_per_scene', 10)} questions/scene
            - **Single speaker:** {dataset_receipt.get('source_dataset', 'LJSpeech speaker LJ (female)')}
            - **{dataset_receipt.get('noise_class_count', 10)} sound classes + Speech**
            - **{dataset_receipt['unique_noise_sources']} unique noise sources**
            - Cross-split source leakage: **{dataset_receipt['cross_split_source_leakage_count']}**
            """
        )
        if "requested_snr_db_values" in dataset_receipt:
            st.markdown(
                f"- Hard SNR levels: **{dataset_receipt['requested_snr_db_values']} dB**\n"
                f"- Difficulty balance: **{dataset_receipt['difficulty_counts']}**"
            )
        if dataset_receipt.get("mean_measured_speech_to_all_interference_snr_db") is not None:
            st.markdown(
                f"- Measured speech/all-noise SNR mean: **{dataset_receipt['mean_measured_speech_to_all_interference_snr_db']:.2f} dB**\n"
                f"- Minimum measured SNR: **{dataset_receipt['min_measured_speech_to_all_interference_snr_db']:.2f} dB**"
            )
    with right:
        st.markdown(
            f"""
            - Detector frame F1 ↑: **{detector.get('frame_f1_↑', float('nan')):.3f}**
            - Detector recall ↑: **{detector['event_recall_↑']:.3f}**
            - Matched detector IoU ↑: **{detector['matched_mean_iou_↑']:.3f}**
            - Transcript đúng nếu WER ≤ 0.25
            - Predicted evidence là **mixture được mask bằng span do model dự đoán**
            - Oracle evidence là tổng các ground-truth stems cần cho câu hỏi
            """
        )

filter_mode = st.radio(
    "Lọc scene theo kết quả",
    ["Tất cả", "Có câu sai", "Tất cả câu đúng"],
    horizontal=True,
    key=f"scene_filter::{variant_name}",
)
scene_options: list[str] = []
for scene in test_scenes:
    rows = results_by_scene[scene["scene_id"]]
    correct_count = sum(bool(row["correct"]) for row in rows)
    if filter_mode == "Có câu sai" and correct_count == len(rows):
        continue
    if filter_mode == "Tất cả câu đúng" and correct_count != len(rows):
        continue
    scene_options.append(scene["scene_id"])
if not scene_options:
    st.warning("Không có scene phù hợp bộ lọc.")
    st.stop()

scene_choices = {
    f"{value} — {sum(bool(row['correct']) for row in results_by_scene[value])}/{len(results_by_scene[value])} câu đúng": value
    for value in scene_options
}
scene_choice = st.selectbox(
    "Chọn test scene",
    list(scene_choices),
    key=f"scene::{variant_name}",
)
scene_id = scene_choices[scene_choice]
scene = scene_by_id[scene_id]
scene_results = results_by_scene[scene_id]
representative = scene_results[0]

st.subheader("1. Audio và timeline")
audio_col, info_col = st.columns([1.05, 1.95])
with audio_col:
    st.caption("Mixture đầu vào")
    st.audio(str(_resolve(scene["mixture_path"])))
    st.markdown(
        f"**Độ dài:** {scene['duration_seconds']:.2f}s  ·  "
        f"**Speaker:** {', '.join(scene.get('speaker_ids', ['LJ']))} / "
        f"{', '.join(scene.get('speaker_groups', ['female']))}  ·  **Events:** {len(scene['events'])}"
    )
    if scene.get("difficulty"):
        noise_line = (
            f"**Difficulty:** `{scene['difficulty']}` · "
            f"**Requested speech/noise SNR:** {scene['requested_speech_to_overlap_noise_snr_db']:+.1f} dB"
        )
        if scene.get("measured_first_speech_to_overlap_noise_snr_db") is not None:
            noise_line += f" · **Measured:** {scene['measured_first_speech_to_overlap_noise_snr_db']:+.2f} dB"
        elif scene.get("measured_speech_to_all_interference_snr_db") is not None:
            noise_line += f" · **Measured speech/all noise:** {scene['measured_speech_to_all_interference_snr_db']:+.2f} dB"
        st.markdown(noise_line)
with info_col:
    gold_rows = []
    for event in sorted(scene["events"], key=lambda event: event["onset_seconds"]):
        gold_rows.append(
            {
                "Vai trò": event["role"],
                "Ground-truth event": event["display_name"],
                "Thời gian": f"{event['onset_seconds']:.2f}–{event['offset_seconds']:.2f}s",
                "Transcript": short_text(event.get("transcript")),
            }
        )
    st.dataframe(pd.DataFrame(gold_rows), hide_index=True, use_container_width=True)

pred_col, asr_col = st.columns(2)
with pred_col:
    st.markdown("**Event inventory do model dự đoán**")
    predicted_rows = [
        {
            "Label": event["label"].replace("_", " "),
            "Span": f"{event['onset_seconds']:.2f}–{event['offset_seconds']:.2f}s",
            "Confidence": f"{event['confidence']:.3f}",
        }
        for event in representative["predicted_events"]
    ]
    st.dataframe(pd.DataFrame(predicted_rows), hide_index=True, use_container_width=True)
with asr_col:
    st.markdown("**Speech spans + Whisper transcript**")
    utterance_rows = [
        {
            "Utterance": index,
            "Predicted span": f"{event['onset_seconds']:.2f}–{event['offset_seconds']:.2f}s",
            "ASR transcript": event["transcript"],
        }
        for index, event in enumerate(representative["predicted_utterances"], start=1)
    ]
    st.dataframe(pd.DataFrame(utterance_rows), hide_index=True, use_container_width=True)

with st.expander("Nghe riêng từng ground-truth source stem", expanded=False):
    stem_choices = {
        f"{event['role']} — {event['display_name']} ({event['onset_seconds']:.2f}–{event['offset_seconds']:.2f}s)": event["event_id"]
        for event in scene["events"]
    }
    stem_choice = st.selectbox(
        "Source event",
        list(stem_choices),
        key=f"stem::{variant_name}::{scene_id}",
    )
    stem_event_id = stem_choices[stem_choice]
    stem_event = next(event for event in scene["events"] if event["event_id"] == stem_event_id)
    st.audio(str(_resolve(stem_event["stem_path"])))

if "cực ồn" in variant_name:
    st.subheader("2. So sánh tách/khử nhiễu lời nói")
    speech_event = next(event for event in scene["events"] if event["label"] == "Speech")
    enhancement_paths = {
        "Mixture gốc": _resolve(scene["mixture_path"]),
        "AudioSep — chọn deploy": AUTOMOTIVE_ENHANCEMENT_OUTPUTS["AudioSep"] / "audio/selected/test" / f"{scene_id}.flac",
        "SepFormer domain": AUTOMOTIVE_ENHANCEMENT_OUTPUTS["SepFormer domain-adapted"] / "audio/selected/test" / f"{scene_id}.flac",
        "Complex mask": AUTOMOTIVE_ENHANCEMENT_OUTPUTS["TF complex mask"] / "audio/selected/test" / f"{scene_id}.flac",
        "Clean upper bound": _resolve(speech_event["stem_path"]),
    }
    audio_columns = st.columns(len(enhancement_paths))
    for column, (label, path) in zip(audio_columns, enhancement_paths.items()):
        with column:
            st.markdown(f"**{label}**")
            st.audio(str(path))
    st.caption(
        "Nghe ưu tiên AudioSep: SepFormer domain có SI-SDRi cao hơn nhưng thường tạo artifact làm ASR sai; "
        "complex mask ít artifact sinh waveform hơn nhưng còn bỏ mất phụ âm ở SNR cực thấp."
    )
    scene_asr = automotive_asr_by_scene.get(scene_id, {})
    def find_asr(suffix: str) -> dict[str, Any] | None:
        return next((value for key, value in scene_asr.items() if key.endswith(suffix)), None)
    asr_display = [
        ("Mixture + oracle speech span", scene_asr.get("mixture")),
        ("AudioSep full", find_asr(":audiosep_full")),
        ("SepFormer domain full", scene_asr.get("SepFormer domain-adapted:domain_enhanced_full")),
        ("Complex mask full", scene_asr.get("TF complex mask:domain_enhanced_full")),
        ("Clean upper bound", find_asr(":clean_upper_bound")),
    ]
    transcript_rows = [{
        "Input cho Whisper-medium": label,
        "Transcript dự đoán": (item or {}).get("hypothesis", "—"),
        "WER ↓": (item or {}).get("wer_↓"),
    } for label, item in asr_display]
    st.markdown(f"**Transcript chuẩn:** {speech_event['transcript']}")
    st.dataframe(pd.DataFrame(transcript_rows).style.format({"WER ↓": "{:.3f}"}, na_rep="—"),
                 hide_index=True, use_container_width=True)

st.subheader("3. Câu hỏi, đáp án và evidence" if "cực ồn" in variant_name else "2. Câu hỏi, đáp án và evidence")
question_choices = {
    f"{'✅' if row['correct'] else '❌'} {row['question_id'].split('_')[-1]} — {short_text(row['question'], 105)}": row
    for row in scene_results
}
question_choice = st.selectbox(
    "Chọn câu hỏi",
    list(question_choices),
    key=f"question::{variant_name}::{scene_id}",
)
result = question_choices[question_choice]
question = question_by_id[result["question_id"]]
st.markdown(f"### {result['question']}")
st.caption(
    f"Input = mixture audio + natural-language question · Parsed operation = "
    f"`{result['parsed_operation']}`"
)
st.info(OPERATION_EXPLANATIONS.get(result["operation"], "Unsupported operation"))

answer_left, answer_right = st.columns(2)
with answer_left:
    st.markdown("**Ground-truth answer**")
    st.markdown(f'<div class="qces-answer">{result["gold_answer"]}</div>', unsafe_allow_html=True)
with answer_right:
    st.markdown(f"**Ours predicted answer — {'✅ đúng' if result['correct'] else '❌ sai'}**")
    st.markdown(f'<div class="qces-answer">{result["predicted_answer"]}</div>', unsafe_allow_html=True)
    if result.get("transcript_wer_↓") is not None:
        st.caption(f"Transcript WER ↓ = {result['transcript_wer_↓']:.3f}; đúng khi WER ≤ 0.25")

af3_result = af3_by_question.get(str(result["question_id"]))
if variant_name.startswith("Hard") and af3_result is not None:
    st.markdown(
        f"**Audio Flamingo 3 + original mixture — "
        f"{'✅ đúng' if af3_result['correct_at_wer_0.25'] else '❌ sai'}**"
    )
    st.markdown(
        f'<div class="qces-answer">{af3_result["prediction"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"AF3 WER ↓ = {af3_result['wer']:.3f}. Input chỉ gồm toàn bộ mixture gốc "
        "+ original question; không dùng output/evidence của ours."
    )

evidence_left, evidence_right = st.columns(2)
with evidence_left:
    st.markdown("**Predicted evidence — from ours**")
    st.audio(str(_resolve(result["predicted_evidence_path"])))
    st.caption(
        "Model tự dự đoán label + span. Audio này là mixture chỉ được giữ tại các span đó; "
        "không dùng ground-truth stem."
    )
with evidence_right:
    st.markdown("**Oracle evidence — upper bound**")
    st.audio(str(_resolve(result["oracle_evidence_path"])))
    st.caption("Tổng các source stems được annotation là cần thiết để kiểm chứng câu hỏi.")

span_rows = [
    {
        "Predicted evidence component": event.get("label", "Speech").replace("_", " "),
        "Span": f"{event['onset_seconds']:.2f}–{event['offset_seconds']:.2f}s",
        "Transcript": short_text(event.get("transcript")),
    }
    for event in result["predicted_evidence_spans"]
]
st.dataframe(pd.DataFrame(span_rows), hide_index=True, use_container_width=True)
st.caption(f"Evidence temporal IoU ↑ của case này: **{result['evidence_iou_↑']:.3f}**")

st.divider()
st.markdown(
    "<span class='small-note'>Demo dùng test split cố định. Mọi predicted answer/evidence đều được "
    "precompute từ detector + ASR; oracle chỉ xuất hiện ở cột đối chiếu bên phải.</span>",
    unsafe_allow_html=True,
)
