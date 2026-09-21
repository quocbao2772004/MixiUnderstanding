#!/usr/bin/env python3
"""Streamlit audit and live Audio -> QCES -> Audio Flamingo 3 demo.

The Streamlit process intentionally uses a lightweight environment.  QCES and
AF3 run in separate, sequential subprocesses so their incompatible dependency
stacks and GPU allocations never coexist inside this process.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.demo_contract import (  # noqa: E402
    compact_scene_events,
    DemoContractError,
    exclusive_lock_available,
    gpu_resource_ready,
    humanize_event_label,
    load_fingerprint_bound_examples,
    load_and_validate_health_receipt,
    relation_requirement,
)


DEFAULT_AF3_MODEL = Path(
    "/home/cuongpv/.cache/huggingface/hub/"
    "models--nvidia--audio-flamingo-3-hf/snapshots/"
    "7d4bae64ee29878af6504ae6f6bb3e40492838ad"
)
DEFAULT_QCES_RUN = (
    PROJECT_ROOT
    / "outputs/qces_v5_overfit30_seed2028/train_refiner_full"
)
DEFAULT_MICROFIT_PREVIEW_ROOT = (
    PROJECT_ROOT
    / "outputs/qces_v5_overfit30_seed2028/eval_refiner_gate"
)
PREVIEW_LABELS = {
    "00_mixture.wav": "Mixture (audio gốc)",
    "01_target_evidence.wav": "Target evidence (oracle)",
    "02_union_evidence.wav": "Union evidence cũ (đã collapse)",
    "03_dual_evidence.wav": "Dual evidence cũ (đã collapse)",
    "04_union_residual.wav": "Union residual cũ",
    "05_dual_residual.wav": "Dual residual cũ",
    "06_target_anchor_role.wav": "Target anchor role",
    "07_target_answer_role.wav": "Target answer role",
}
BEATS_PREVIEW_LABELS = {
    "00_mixture.wav": "Mixture (audio gốc)",
    "01_oracle_evidence.wav": "Oracle evidence E*",
    "02_oracle_residual.wav": "Oracle residual R*",
}
MICROFIT_PREVIEW_LABELS = {
    "mixture.wav": "Mixture X",
    "predicted_evidence.wav": "Predicted evidence E",
    "target_evidence.wav": "Target evidence E*",
    "predicted_residual.wav": "Predicted residual R",
    "target_residual.wav": "Target residual R*",
}
AUDIBILITY_PACKET_LABELS = {
    "00_original_mixture.wav": "Mixture cũ · random crop",
    "01_random_crop_meow_stem.wav": "Meow stem cũ · random crop",
    "02_max_variance_meow_stem.wav": "Meow stem diagnostic · max variance",
    "03_max_variance_mixture.wav": "Mixture A/B diagnostic · thay Meow crop",
}
GPU_QUEUE_STAGES = (
    "microfit",
    "cee_memory",
    "cee_pilot",
    "phi_forward",
    "phi_oracle",
    "caption_baseline",
    "ablation_screen",
)
PREVIEW_CASE_IDS = {
    "case_00_after": "val_000010_1_00",
    "case_01_before": "val_000010_2_04",
    "case_02_first": "val_000010_1_10",
    "case_03_no_evidence": "val_000017_2_07",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_devpilot_preview_seed2026/listen",
    )
    parser.add_argument(
        "--beats-preview-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_beats_oracle_val_seed2026/listening",
    )
    parser.add_argument(
        "--microfit-preview-root",
        type=Path,
        default=DEFAULT_MICROFIT_PREVIEW_ROOT,
    )
    parser.add_argument(
        "--audibility-report",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_audibility_audit_20260723/audibility_report.json",
    )
    parser.add_argument(
        "--audibility-listening-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_audibility_audit_20260723/listening",
    )
    parser.add_argument(
        "--semantic-crop-bank",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_semantic_crop_bank_v2_20260723/crop_bank.json",
    )
    parser.add_argument(
        "--semantic-crop-listening-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_semantic_crop_bank_v2_20260723/listening",
    )
    parser.add_argument(
        "--realdesed-root",
        type=Path,
        default=PROJECT_ROOT / "data/qces_realdesed_v1_dev",
    )
    parser.add_argument(
        "--realdesed-audit-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_realdesed_v1_dev_audit_20260723",
    )
    parser.add_argument(
        "--qces-checkpoint",
        type=Path,
        default=DEFAULT_QCES_RUN / "checkpoint.pt",
    )
    parser.add_argument(
        "--health-receipt",
        type=Path,
        default=DEFAULT_QCES_RUN / "demo_health_receipt.json",
    )
    parser.add_argument("--af3-model", type=Path, default=DEFAULT_AF3_MODEL)
    parser.add_argument(
        "--qces-python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/comfyui/bin/python"),
    )
    parser.add_argument(
        "--audioqa-python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/qces-sam/bin/python"),
    )
    parser.add_argument(
        "--audiosep-root", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep"
    )
    parser.add_argument(
        "--audiosep-config",
        type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep/config/audiosep_base.yaml",
    )
    parser.add_argument(
        "--audiosep-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "code/baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=PROJECT_ROOT / "outputs/qces_streamlit_demo"
    )
    parser.add_argument(
        "--queue-status",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_gpu_queue_status.json",
    )
    parser.add_argument(
        "--gpu-workflow-lock",
        type=Path,
        default=PROJECT_ROOT / "outputs/.qces_gpu_queue.lock",
    )
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=7_000)
    parser.add_argument("--maximum-gpu-utilization-percent", type=int, default=10)
    args, _ = parser.parse_known_args(argv)
    return args


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _gpu_state() -> dict[str, int] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first = completed.stdout.strip().splitlines()[0]
        free_text, utilization_text, temperature_text = [
            piece.strip() for piece in first.split(",")
        ]
        return {
            "free_memory_mib_↑": int(free_text),
            "utilization_percent_↓": int(utilization_text),
            "temperature_celsius_↓": int(temperature_text),
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(CODE_ROOT) if not existing else str(CODE_ROOT) + os.pathsep + existing
    )
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    return environment


def _run(
    command: Sequence[str], timeout_seconds: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=_command_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


@contextlib.contextmanager
def _exclusive_demo_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "một inference khác đang chạy; đợi lượt đó xong"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _tail(text: str, limit: int = 6_000) -> str:
    return text if len(text) <= limit else "…" + text[-limit:]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


@st.cache_data(show_spinner=False)
def _read_audibility_report(
    path_text: str, modified_time_ns: int
) -> dict[str, Any]:
    # modified_time_ns is deliberately part of the cache key.
    del modified_time_ns
    return _read_json(Path(path_text))


@st.cache_data(show_spinner=False)
def _read_jsonl(path_text: str, modified_time_ns: int) -> list[dict[str, Any]]:
    del modified_time_ns
    rows: list[dict[str, Any]] = []
    with Path(path_text).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(payload)
    return rows


def _scene_audibility(
    report_path: Path, scene_id: object
) -> dict[str, Mapping[str, Any]]:
    if not report_path.is_file() or not isinstance(scene_id, str):
        return {}
    report = _read_audibility_report(
        str(report_path.resolve()), report_path.stat().st_mtime_ns
    )
    events = report.get("events", [])
    if not isinstance(events, list):
        return {}
    return {
        str(event["event_id"]): event
        for event in events
        if isinstance(event, Mapping)
        and event.get("scene_id") == scene_id
        and isinstance(event.get("event_id"), str)
    }


def _safe_child(root: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        return None
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


@st.cache_data(show_spinner=False)
def _audio_excerpt(
    path_text: str,
    start_seconds: float,
    end_seconds: float,
    normalize_for_inspection: bool,
    modified_time_ns: int,
) -> tuple[bytes, dict[str, float | bool]]:
    # modified_time_ns is part of the cache key so regenerated audio is not stale.
    del modified_time_ns
    with wave.open(path_text, "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        compression = reader.getcomptype()
        frames = reader.readframes(frame_count)
    if compression != "NONE" or channels <= 0 or sample_rate <= 0:
        raise ValueError("only uncompressed PCM WAV is supported")
    if sample_width == 1:
        decoded = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        decoded = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        triplets = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        integers = (
            triplets[:, 0].astype(np.int32)
            | (triplets[:, 1].astype(np.int32) << 8)
            | (triplets[:, 2].astype(np.int32) << 16)
        )
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        decoded = integers.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        decoded = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width}")
    if decoded.size % channels:
        raise ValueError("PCM frame/channel count mismatch")
    mono = decoded.reshape(-1, channels).mean(axis=1, dtype=np.float64)
    start = max(0, min(len(mono), int(round(start_seconds * sample_rate))))
    stop = max(start, min(len(mono), int(round(end_seconds * sample_rate))))
    excerpt = mono[start:stop].astype(np.float32, copy=True)
    if not len(excerpt) or not np.isfinite(excerpt).all():
        raise ValueError("audio excerpt is empty or non-finite")
    rms = float(np.sqrt(np.mean(np.square(excerpt, dtype=np.float64))))
    peak = float(np.max(np.abs(excerpt)))
    rms_dbfs = -120.0 if rms <= 1e-6 else 20.0 * np.log10(rms)
    peak_dbfs = -120.0 if peak <= 1e-6 else 20.0 * np.log10(peak)
    applied_gain_db = 0.0
    limited_sample_fraction = 0.0
    if normalize_for_inspection and rms > 1e-6:
        # Bring the complete crop to -16 dBFS RMS, then limit only samples that
        # would clip. This is deliberately louder than peak normalization for
        # sparse events such as Meow; it is an inspection rendering, never a
        # model input or an SNR measurement.
        gain = min(100.0, 10.0 ** ((-16.0 - rms_dbfs) / 20.0))
        excerpt *= gain
        applied_gain_db = 20.0 * np.log10(gain)
        limited_sample_fraction = float(np.mean(np.abs(excerpt) > 0.98))
        excerpt = np.clip(excerpt, -0.98, 0.98)
    buffer = io.BytesIO()
    pcm = np.round(np.clip(excerpt, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())
    return buffer.getvalue(), {
        "duration_seconds": len(excerpt) / sample_rate,
        "original_rms_dbfs ↑": float(rms_dbfs),
        "original_peak_dbfs ↑": float(peak_dbfs),
        "inspection_gain_db": float(applied_gain_db),
        "limited_sample_fraction ↓": limited_sample_fraction,
        "normalized_for_inspection": normalize_for_inspection,
    }


def _format_intervals(intervals: object) -> str:
    if not isinstance(intervals, Sequence) or isinstance(intervals, (str, bytes)):
        return "không có"
    rendered: list[str] = []
    for interval in intervals:
        if (
            not isinstance(interval, Sequence)
            or isinstance(interval, (str, bytes))
            or len(interval) != 2
        ):
            continue
        try:
            start, end = float(interval[0]), float(interval[1])
        except (TypeError, ValueError):
            continue
        rendered.append(f"{start:.2f}–{end:.2f}s")
    return ", ".join(rendered) if rendered else "không có"


def _event_names(events: Sequence[Mapping[str, Any]], ids: object) -> str:
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        return "không có"
    wanted = {str(value) for value in ids if isinstance(value, str)}
    names = [
        f"{event['sound']} #{event['occurrence']}"
        for event in events
        if event.get("event_id") in wanted
    ]
    return " + ".join(names) if names else "không có"


def _render_event_inventory(
    events: Sequence[Mapping[str, Any]], *, predicted: bool
) -> None:
    if predicted:
        st.warning(
            "Đây là danh sách AF3 **dự đoán khi nghe mixture**, không phải nhãn "
            "ground truth; nó có thể bỏ sót hoặc gọi sai âm."
        )
        rows = [
            {"thứ tự": event.get("order"), "âm nghe được (dự đoán)": event.get("sound")}
            for event in events
        ]
    else:
        st.warning(
            "Đây là **công thức trộn synthetic (recipe)**, không phải danh sách "
            "những âm đã được xác nhận là nghe rõ. Một stem có thể tồn tại nhưng "
            "bị âm khác che trong mixture."
        )
        rows = [
            {
                "thứ tự onset": index,
                "stem trong recipe": f"{event['sound']} #{event['occurrence']}",
                "thời gian": (
                    f"{float(event['start_seconds']):.2f}–"
                    f"{float(event['end_seconds']):.2f}s"
                ),
                "loại": (
                    "âm trong timeline"
                    if event.get("kind") == "semantic"
                    else "âm nền/nuisance"
                ),
            }
            for index, event in enumerate(events, 1)
        ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có danh sách âm đáng tin cậy cho cửa sổ này.")


def _load_preview_story(args: argparse.Namespace, case: Path) -> dict[str, Any] | None:
    item_id = PREVIEW_CASE_IDS.get(case.name)
    report_path = args.preview_root.parent / "union_single" / "evaluation_report.json"
    if item_id is None or not report_path.is_file():
        return None
    report = _read_json(report_path)
    item = next(
        (
            value
            for value in report.get("items", [])
            if isinstance(value, Mapping) and value.get("id") == item_id
        ),
        None,
    )
    manifest_text = report.get("manifest")
    if item is None or not isinstance(manifest_text, str):
        return None
    manifest = Path(manifest_text).resolve()
    if not manifest.is_file():
        return None
    row = None
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            if isinstance(candidate, Mapping) and candidate.get("id") == item_id:
                row = candidate
                break
    if row is None:
        return None
    return {
        "item": dict(item),
        "row": dict(row),
        "events": compact_scene_events(row.get("events")),
        "manifest_root": manifest.parent,
    }


def _asset_readiness(
    args: argparse.Namespace,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    errors: list[str] = []
    health: dict[str, Any] | None = None
    required = {
        "QCES Python": args.qces_python,
        "AudioQA Python": args.audioqa_python,
        "AudioSep source": args.audiosep_root,
        "AudioSep config": args.audiosep_config,
        "AudioSep checkpoint": args.audiosep_checkpoint,
        "AF3 model": args.af3_model,
    }
    for label, path in required.items():
        if not path.exists():
            errors.append(f"thiếu {label}: {path}")
    try:
        health = load_and_validate_health_receipt(
            args.qces_checkpoint.resolve(), args.health_receipt.resolve()
        )
    except (DemoContractError, OSError) as error:
        errors.append(f"checkpoint chưa qua anti-collapse gate: {error}")
    return not errors, errors, health


def render_preview(
    args: argparse.Namespace, *, include_additional_audits: bool = True
) -> None:
    st.subheader("Case setup và oracle hoạt động như thế nào?")
    st.error(
        "**KHÔNG PHẢI KẾT QUẢ HIỆN TẠI:** checkpoint dev-pilot cũ đã collapse "
        "và sinh evidence gần như im lặng. Tab này mặc định chỉ minh hoạ input, "
        "yêu cầu và oracle; output lỗi được chuyển xuống phần audit kỹ thuật."
    )
    cases = sorted(path for path in args.preview_root.glob("case_*") if path.is_dir())
    if not cases:
        st.error(f"Không thấy preview cases tại {args.preview_root}")
        return
    case = st.selectbox(
        "Chọn tình huống",
        cases,
        format_func=lambda path: path.name.replace("case_", "").replace("_", " · "),
        key=(
            "preview_case_technical"
            if include_additional_audits
            else "preview_case_story"
        ),
    )
    story = _load_preview_story(args, case)
    if story is None:
        st.error("Không ghép được case với manifest/report đã dùng để tạo audio.")
        return
    item = story["item"]
    row = story["row"]
    events = story["events"]
    audibility_by_event = _scene_audibility(
        args.audibility_report, row.get("scene_id")
    )

    st.markdown("### 1 · Mixture và công thức trộn")
    st.audio(str(case / "00_mixture.wav"), format="audio/wav")
    _render_event_inventory(events, predicted=False)

    if audibility_by_event:
        audibility_rows = []
        for event in events:
            metrics = audibility_by_event.get(str(event.get("event_id")))
            if metrics is None:
                continue
            band = str(metrics.get("screening_band"))
            audibility_rows.append(
                {
                    "stem": f"{event['sound']} #{event['occurrence']}",
                    "QA role": (
                        "+".join(metrics.get("roles_across_scene_questions", []))
                        or "không"
                    ),
                    "stem RMS dBFS ↑": round(
                        float(metrics["interval_stem_rms_dbfs ↑"]), 2
                    ),
                    "event/background SNR dB ↑": round(
                        float(metrics["event_to_background_snr_db ↑"]), 2
                    ),
                    "activity fraction ↑": round(
                        float(metrics["within_interval_activity_fraction ↑"]), 3
                    ),
                    "acoustic screen": {
                        "high_masking_or_low_level_risk": "🔴 high risk",
                        "manual_review": "🟠 cần nghe lại",
                        "lower_risk_candidate": "🟢 lower risk",
                    }.get(band, band),
                }
            )
        st.dataframe(audibility_rows, use_container_width=True, hide_index=True)
        st.caption(
            "SNR ↑ càng cao càng dễ nổi trên phần audio còn lại. Đây chỉ là "
            "acoustic screen; nó không chứng minh stem mang đúng nhãn hay con "
            "người nhận ra được."
        )

    raw_events = row.get("events", [])
    if isinstance(raw_events, list) and raw_events:
        with st.expander(
            "Nghe đúng interval của từng event trên timeline scene",
            expanded=True,
        ):
            selected_event = st.selectbox(
                "Event trong recipe",
                raw_events,
                format_func=lambda event: (
                    f"{humanize_event_label(str(event.get('label', 'unknown')))} "
                    f"#{event.get('occurrence_index', '?')} · "
                    f"{float(event.get('onset_seconds', 0.0)):.2f}–"
                    f"{float(event.get('offset_seconds', 0.0)):.2f}s"
                ),
                key=f"recipe_event_{case.name}_{include_additional_audits}",
            )
            onset = float(selected_event.get("onset_seconds", 0.0))
            offset = float(selected_event.get("offset_seconds", onset))
            stem_path = (
                Path(story["manifest_root"]) / str(selected_event.get("stem_path", ""))
            ).resolve()
            if stem_path.is_file():
                st.success(
                    f"Đang cắt đúng **{onset:.2f}–{offset:.2f}s** trên timeline "
                    f"scene cho {humanize_event_label(str(selected_event.get('label')))} "
                    f"#{selected_event.get('occurrence_index')}. Hai player crop "
                    "bên trái cùng lấy chính xác interval này."
                )
                interval_players = st.columns(3)
                boosted, boosted_stats = _audio_excerpt(
                    str(stem_path),
                    onset,
                    offset,
                    True,
                    stem_path.stat().st_mtime_ns,
                )
                true_level, true_stats = _audio_excerpt(
                    str(stem_path),
                    onset,
                    offset,
                    False,
                    stem_path.stat().st_mtime_ns,
                )
                mixture_path = case / "00_mixture.wav"
                mixture_crop, _ = _audio_excerpt(
                    str(mixture_path),
                    onset,
                    offset,
                    False,
                    mixture_path.stat().st_mtime_ns,
                )
                with interval_players[0]:
                    st.markdown("**A · Event crop tăng loudness để nhận diện**")
                    st.audio(boosted, format="audio/wav")
                    st.caption(
                        f"Scene {onset:.2f}–{offset:.2f}s → player 0.00–"
                        f"{offset - onset:.2f}s. Boost "
                        f"{boosted_stats['inspection_gain_db']:+.1f} dB; "
                        f"limited {100.0 * float(boosted_stats['limited_sample_fraction ↓']):.2f}% ↓. "
                        "Chỉ để nghe nội dung."
                    )
                with interval_players[1]:
                    st.markdown("**B · Cùng event crop, đúng level scene**")
                    st.audio(true_level, format="audio/wav")
                    st.caption(
                        f"RMS {true_stats['original_rms_dbfs ↑']:.1f} dBFS ↑ · "
                        f"peak {true_stats['original_peak_dbfs ↑']:.1f} dBFS ↑."
                    )
                with interval_players[2]:
                    st.markdown("**C · Mixture cắt đúng cùng interval**")
                    st.audio(mixture_crop, format="audio/wav")
                    st.caption(
                        f"Chỉ phần mixture scene {onset:.2f}–{offset:.2f}s; "
                        "so với A/B để phát hiện masking."
                    )

                st.markdown(
                    f"**D · Full stem 10 giây:** signal phải chỉ xuất hiện tại "
                    f"`{onset:.2f}–{offset:.2f}s`; ngoài interval là zero."
                )
                st.audio(str(stem_path), format="audio/wav")
                st.warning(
                    "Nếu player A đã được boost mà vẫn không nhận ra đúng nhãn "
                    f"`{humanize_event_label(str(selected_event.get('label')))}`, "
                    "đó là **source/crop semantic mismatch của dataset**. Không "
                    "được giải thích nó là model nghe kém hoặc chỉ tăng gain rồi "
                    "giữ sample."
                )
                selected_metrics = audibility_by_event.get(
                    str(selected_event.get("event_id"))
                )
                if selected_metrics is not None:
                    st.dataframe(
                        [
                            {
                                "metric": "Stem RMS dBFS ↑",
                                "value": round(
                                    float(
                                        selected_metrics[
                                            "interval_stem_rms_dbfs ↑"
                                        ]
                                    ),
                                    2,
                                ),
                            },
                            {
                                "metric": "Event/background SNR dB ↑",
                                "value": round(
                                    float(
                                        selected_metrics[
                                            "event_to_background_snr_db ↑"
                                        ]
                                    ),
                                    2,
                                ),
                            },
                            {
                                "metric": "Within-interval activity ↑",
                                "value": round(
                                    float(
                                        selected_metrics[
                                            "within_interval_activity_fraction ↑"
                                        ]
                                    ),
                                    3,
                                ),
                            },
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                st.caption(
                    "A và B luôn được cắt từ rendered stem bằng chính onset/offset "
                    "đang hiển thị, không dùng timestamp của source gốc. D giữ "
                    "nguyên hệ thời gian 10 giây để kiểm tra alignment."
                )
            else:
                st.warning(f"Không tìm thấy stem: {stem_path}")

    audibility_packet = (
        args.audibility_listening_root
        / f"{row.get('scene_id')}_meow"
    )
    if audibility_packet.is_dir():
        with st.expander("A/B lỗi Meow: random crop cũ so với max-variance diagnostic"):
            st.info(
                "Đây là diagnostic data, chưa phải model output và chưa thay vào "
                "benchmark. Chỉ stem Meow được đổi; mọi âm khác giữ nguyên."
            )
            st.warning(
                "Max-variance sửa được Meow này (`BEATs 0.005 → 0.822 ↑`) nhưng "
                "**đã bị loại làm policy toàn cục**: trên 238 semantic event "
                "validation nó chỉ cải thiện 53.78% ↑ và làm 16.39% ↓ event "
                "giảm ít nhất 0.05."
            )
            packet_paths = [
                audibility_packet / name for name in AUDIBILITY_PACKET_LABELS
            ]
            packet_columns = st.columns(2)
            for index, path in enumerate(packet_paths):
                with packet_columns[index % 2]:
                    st.markdown(f"**{AUDIBILITY_PACKET_LABELS[path.name]}**")
                    st.audio(str(path), format="audio/wav")
            st.caption(
                "Nghe 01 → 02 để kiểm tra crop; sau đó 00 → 03 để kiểm tra "
                "Meow có nổi lên trong mixture hay vẫn bị masking."
            )

    st.markdown("### 2 · Câu hỏi yêu cầu tìm gì?")
    st.info(f"**Câu hỏi:** {item['question']}")
    st.markdown(f"**Yêu cầu:** {relation_requirement(item.get('relation'), item['question'])}")

    st.markdown("### 3 · Evidence phải chứa gì?")
    if bool(item.get("no_evidence")):
        st.success(
            "Anchor được hỏi không tồn tại đủ trong scene → đáp án đúng là "
            "`no_evidence`, evidence chuẩn phải im lặng."
        )
    else:
        explanation_columns = st.columns(2)
        explanation_columns[0].info(
            "**Anchor cần giữ**\n\n"
            f"{_event_names(events, row.get('anchor_event_ids', []))}\n\n"
            f"{_format_intervals(row.get('anchor_intervals'))}"
        )
        explanation_columns[1].info(
            "**Answer event cần giữ**\n\n"
            f"{_event_names(events, row.get('answer_event_ids', []))}\n\n"
            f"{_format_intervals(row.get('answer_intervals'))}"
        )

    st.markdown("### 4 · Nghe oracle: evidence đúng phải giữ gì?")
    main_audio = (
        ("Audio gốc", case / "00_mixture.wav"),
        ("Evidence chuẩn E*", case / "01_target_evidence.wav"),
        ("Anchor chuẩn", case / "06_target_anchor_role.wav"),
        ("Answer chuẩn", case / "07_target_answer_role.wav"),
    )
    columns = st.columns(4)
    for column, (label, path) in zip(columns, main_audio):
        with column:
            st.markdown(f"**{label}**")
            st.audio(str(path), format="audio/wav")

    with st.expander(
        "Output checkpoint CŨ ĐÃ FAIL · chỉ mở để chẩn đoán collapse"
    ):
        st.error(
            "Các file predicted dưới đây gần như im lặng là một failure đã biết; "
            "không dùng chúng để đánh giá hướng QCES hiện tại."
        )
        st.dataframe(
            [
                {
                    "metric": "Temporal IoU ↑",
                    "value": item.get("temporal_iou"),
                },
                {
                    "metric": "Evidence SD-SDRi dB ↑",
                    "value": item.get("evidence_sd_sdri"),
                },
                {
                    "metric": "Predicted retained ratio ↑ (case answerable)",
                    "value": item.get("retained_ratio"),
                },
                {
                    "metric": "Target retained ratio",
                    "value": item.get("target_retained_ratio"),
                },
            ],
            use_container_width=True,
            hide_index=True,
        )
        predicted_intervals = item.get("predicted_evidence_intervals", [])
        if predicted_intervals:
            st.markdown(
                "**Checkpoint cũ tìm được:** "
                + _format_intervals(predicted_intervals)
            )
        elif bool(item.get("no_evidence")):
            st.markdown(
                "**Checkpoint cũ giữ lại:** không có interval, nhưng đầu abstention "
                f"chỉ cho `P(no evidence)={float(item['no_evidence_probability']):.3f}` "
                "(< 0.5), nên vẫn bị tính là sai."
            )
        else:
            st.error(
                "Checkpoint cũ tìm được: **không có interval nào** → evidence collapse."
            )
        technical_wavs = [
            path
            for path in sorted(case.glob("*.wav"))
            if path.name not in {
                "00_mixture.wav",
                "01_target_evidence.wav",
                "06_target_anchor_role.wav",
                "07_target_answer_role.wav",
            }
        ]
        for left_index in range(0, len(technical_wavs), 2):
            technical_columns = st.columns(2)
            for column, path in zip(
                technical_columns, technical_wavs[left_index : left_index + 2]
            ):
                with column:
                    st.markdown(f"**{PREVIEW_LABELS.get(path.name, path.name)}**")
                    st.audio(str(path), format="audio/wav")
    readme = args.preview_root / "README.md"
    if readme.is_file():
        with st.expander("Toàn bộ protocol audit"):
            st.markdown(readme.read_text(encoding="utf-8"))

    if not include_additional_audits:
        return

    st.divider()
    st.subheader("Healthy microfit checkpoint · predicted vs target")
    try:
        microfit_health = load_and_validate_health_receipt(
            args.qces_checkpoint.resolve(), args.health_receipt.resolve()
        )
    except (DemoContractError, OSError) as error:
        st.info(
            "Packet này chỉ xuất hiện sau khi checkpoint qua toàn bộ anti-collapse "
            f"gate: {error}"
        )
    else:
        report_path = (args.microfit_preview_root / "evaluation_report.json").resolve()
        authorized_report_path = Path(
            microfit_health["evaluation_report"]["path"]
        ).resolve()
        if report_path != authorized_report_path:
            st.error(
                "Microfit preview root không trỏ tới report đã được health receipt "
                f"bind: {authorized_report_path}"
            )
            return
        if not report_path.is_file():
            st.error(f"Health receipt tồn tại nhưng thiếu report: {report_path}")
        else:
            report = _read_json(report_path)
            rendering = report.get("audio_rendering", {})
            requested_ids = set(rendering.get("requested_item_ids", []))
            items = [
                item
                for item in report.get("items", [])
                if isinstance(item, Mapping) and item.get("id") in requested_ids
            ]
            if not items:
                st.error("Health report không có listening item đã pre-register.")
            else:
                selected = st.selectbox(
                    "Healthy checkpoint case",
                    items,
                    format_func=lambda item: (
                        f"{item['relation']} | "
                        f"{'no-evidence' if item['no_evidence'] else 'answerable'} | "
                        f"{item['id']}"
                    ),
                    key="healthy_microfit_case",
                )
                st.markdown(f"**Question:** {selected['question']}")
                st.caption(
                    "Nghe theo thứ tự: X → so E với E* → chỉ mở R/R* nếu cần "
                    "kiểm tra leakage."
                )
                st.dataframe(
                    [
                        {
                            "metric": "Temporal IoU ↑",
                            "value": selected.get("temporal_iou"),
                        },
                        {
                            "metric": "Evidence SD-SDRi ↑",
                            "value": selected.get("evidence_sd_sdri"),
                        },
                        {
                            "metric": "Mixture consistency L1 ↓",
                            "value": selected.get("mixture_consistency_l1"),
                        },
                        {
                            "metric": "Evidence retained ratio ↓",
                            "value": selected.get("retained_ratio"),
                        },
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                scene_dir = args.microfit_preview_root / str(selected["scene_id"])
                question_dir = scene_dir / (
                    f"q{selected['question_index']}_{selected['question_type']}"
                )
                primary_paths = (
                    scene_dir / "mixture.wav",
                    question_dir / "predicted_evidence.wav",
                    question_dir / "target_evidence.wav",
                )
                primary_columns = st.columns(3)
                for column, path in zip(primary_columns, primary_paths):
                    with column:
                        st.markdown(f"**{MICROFIT_PREVIEW_LABELS[path.name]}**")
                        st.audio(str(path), format="audio/wav")
                with st.expander("Residual leakage check · R so với R*"):
                    residual_columns = st.columns(2)
                    residual_paths = (
                        question_dir / "predicted_residual.wav",
                        question_dir / "target_residual.wav",
                    )
                    for column, path in zip(residual_columns, residual_paths):
                        with column:
                            st.markdown(f"**{MICROFIT_PREVIEW_LABELS[path.name]}**")
                            st.audio(str(path), format="audio/wav")

    st.divider()
    st.subheader("BEATs oracle · kiểm tra giới hạn class/instance")
    st.info(
        "Đây là oracle data/auditor audit, chưa phải predicted QCES. Nghe để "
        "phân biệt leakage thật với residual còn một event hợp lệ cùng class."
    )
    beats_cases = sorted(
        path for path in args.beats_preview_root.glob("case_*") if path.is_dir()
    )
    if not beats_cases:
        st.warning(f"Không thấy BEATs listening packet tại {args.beats_preview_root}")
        return
    beats_case = st.selectbox(
        "BEATs case",
        beats_cases,
        format_func=lambda path: path.name,
        key="beats_oracle_preview_case",
    )
    beats_wavs = sorted(beats_case.glob("*.wav"))
    beats_columns = st.columns(len(beats_wavs))
    for column, path in zip(beats_columns, beats_wavs):
        with column:
            st.markdown(f"**{BEATS_PREVIEW_LABELS.get(path.name, path.name)}**")
            st.caption(path.name)
            st.audio(str(path), format="audio/wav")
    beats_readme = args.beats_preview_root / "README.md"
    if beats_readme.is_file():
        with st.expander("BEATs: nghe case nào và diễn giải ra sao"):
            st.markdown(beats_readme.read_text(encoding="utf-8"))


def render_sound_explorer(args: argparse.Namespace) -> None:
    st.subheader("Duyệt từng loại âm và từng crop thật trong QCES-v5")
    st.info(
        "Dùng player **crop normalize** để kiểm tra source/crop có đúng semantic. "
        "Sau đó nghe **true level** và **mixture** để kiểm tra âm có bị che. "
        "Không dùng audio normalize để đánh giá SNR hoặc chất lượng separator."
    )
    crop_bank_path = args.semantic_crop_bank.resolve()
    if crop_bank_path.is_file():
        try:
            crop_bank = _read_audibility_report(
                str(crop_bank_path), crop_bank_path.stat().st_mtime_ns
            )
            crop_entries = [
                entry
                for entry in crop_bank.get("entries", [])
                if isinstance(entry, Mapping)
            ]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            st.error(f"Không đọc được semantic crop bank mới: {error}")
            crop_entries = []
        if crop_entries:
            st.success(
                "Crop bank theo nhãn đã sẵn sàng. Đây là cửa sổ sẽ dùng cho "
                "artifact rebuild; BEATs chỉ là data curator, không phải final auditor."
            )
            with st.expander("Nghe crop bank mới: SELECTED so với RUNNER-UP", expanded=True):
                listening_root = args.semantic_crop_listening_root.resolve()
                queue_path = listening_root / "listening_queue.jsonl"
                decision_path = listening_root / "semantic_crop_decisions.json"
                queue_rows = (
                    _read_jsonl(str(queue_path), queue_path.stat().st_mtime_ns)
                    if queue_path.is_file()
                    else []
                )
                bank_index = {
                    (str(entry["source_id"]), str(entry["label"])): entry
                    for entry in crop_entries
                }
                decisions = _read_json(decision_path) if decision_path.is_file() else {}
                queue_mode = st.radio(
                    "Phạm vi nghe",
                    ("80 case bắt buộc", "Toàn bộ 339 source"),
                    horizontal=True,
                    key="semantic_crop_bank_scope",
                    disabled=not queue_rows,
                )
                queue_row: Mapping[str, Any] | None = None
                if queue_mode == "80 case bắt buộc" and queue_rows:
                    reviewed = sum(
                        isinstance(decisions.get(f"{row['label']}::{row['source_id']}"), Mapping)
                        for row in queue_rows
                    )
                    status_columns = st.columns(4)
                    status_columns[0].metric("Đã nghe ↑", reviewed)
                    status_columns[1].metric("Còn lại ↓", len(queue_rows) - reviewed)
                    status_columns[2].metric(
                        "Pass ↑",
                        sum(
                            decisions.get(f"{row['label']}::{row['source_id']}", {}).get("decision")
                            == "pass"
                            for row in queue_rows
                        ),
                    )
                    status_columns[3].metric(
                        "Reject/unsure ↓",
                        sum(
                            decisions.get(f"{row['label']}::{row['source_id']}", {}).get("decision")
                            in {"reject", "unsure"}
                            for row in queue_rows
                        ),
                    )
                    def queue_case_label(row: Mapping[str, Any]) -> str:
                        row_key = f"{row['label']}::{row['source_id']}"
                        row_status = decisions.get(row_key, {}).get(
                            "decision", "CHƯA NGHE"
                        )
                        return (
                            f"#{int(row['queue_index']):03d} · "
                            f"{humanize_event_label(str(row['label']))} · "
                            f"source {row['source_id']} · {row_status}"
                        )

                    queue_row = st.selectbox(
                        "Case cần duyệt",
                        queue_rows,
                        format_func=queue_case_label,
                        key="semantic_crop_queue_case",
                    )
                    bank_entry = bank_index[
                        (str(queue_row["source_id"]), str(queue_row["label"]))
                    ]
                    bank_label = str(bank_entry["label"])
                else:
                    bank_labels = sorted({str(entry["label"]) for entry in crop_entries})
                    bank_default = "Meow" if "Meow" in bank_labels else bank_labels[0]
                    bank_label = st.selectbox(
                        "Loại âm trong crop bank mới",
                        bank_labels,
                        index=bank_labels.index(bank_default),
                        format_func=humanize_event_label,
                        key="semantic_crop_bank_label",
                    )
                    bank_class_entries = sorted(
                        [entry for entry in crop_entries if entry.get("label") == bank_label],
                        key=lambda entry: (
                            float(entry["selected"]["label_probability ↑"]),
                            str(entry["source_id"]),
                        ),
                    )
                    bank_entry = st.selectbox(
                        f"Source của {humanize_event_label(bank_label)}",
                        bank_class_entries,
                        format_func=lambda entry: (
                            f"source {entry['source_id']} · selected score "
                            f"{float(entry['selected']['label_probability ↑']):.4f} ↑ · "
                            f"margin {float(entry['selection_margin ↑']):.4f} ↑"
                        ),
                        key="semantic_crop_bank_source",
                    )
                source_path = _safe_child(PROJECT_ROOT, bank_entry.get("source_path"))
                bank_columns = st.columns(2)
                for column, role, title in (
                    (bank_columns[0], "selected", "SELECTED · crop sẽ dùng"),
                    (bank_columns[1], "runner_up", "RUNNER-UP · đối chiếu"),
                ):
                    with column:
                        candidate = bank_entry[role]
                        interval = candidate["crop_interval_seconds"]
                        st.markdown(f"**{title}**")
                        st.caption(
                            f"Source time {float(interval[0]):.2f}–"
                            f"{float(interval[1]):.2f}s · BEATs label score "
                            f"{float(candidate['label_probability ↑']):.4f} ↑"
                        )
                        if source_path is None:
                            st.error("Thiếu source audio của crop bank.")
                        else:
                            payload, _ = _audio_excerpt(
                                str(source_path),
                                float(interval[0]),
                                float(interval[1]),
                                True,
                                source_path.stat().st_mtime_ns,
                            )
                            st.audio(payload, format="audio/wav")
                st.caption(
                    "Việc cần nghe: SELECTED có nhận ra đúng loại âm đang hiển thị "
                    "không. Chỉ so RUNNER-UP nếu SELECTED mơ hồ. Audio ở đây được "
                    "normalize để soi semantic, không phải level model nhận."
                )
                if queue_row is not None:
                    decision_key = f"{bank_label}::{bank_entry['source_id']}"
                    previous_decision = decisions.get(decision_key, {})
                    decision_options = ("pass", "reject", "unsure")
                    previous_value = previous_decision.get("decision", "unsure")
                    decision = st.radio(
                        "Kết luận SELECTED",
                        decision_options,
                        index=(
                            decision_options.index(previous_value)
                            if previous_value in decision_options
                            else 2
                        ),
                        horizontal=True,
                        key=f"semantic_crop_decision_{decision_key}",
                        help="pass = nhận ra đúng label; reject = sai/không có; unsure = chưa chắc.",
                    )
                    note = st.text_input(
                        "Ghi chú ngắn",
                        value=str(previous_decision.get("note", "")),
                        key=f"semantic_crop_note_{decision_key}",
                    )
                    if st.button(
                        "Lưu kết luận case này",
                        key=f"semantic_crop_save_{decision_key}",
                        type="primary",
                    ):
                        latest = _read_json(decision_path) if decision_path.is_file() else {}
                        latest[decision_key] = {
                            "decision": decision,
                            "note": note.strip(),
                            "label": bank_label,
                            "source_id": str(bank_entry["source_id"]),
                            "selected_crop_interval_seconds": bank_entry["selected"][
                                "crop_interval_seconds"
                            ],
                            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        _json_write(decision_path, latest)
                        st.success("Đã lưu. Chuyển sang case tiếp theo trong selectbox.")
        else:
            st.warning("Semantic crop bank tồn tại nhưng chưa có entry hợp lệ.")
    else:
        st.info(
            "Semantic crop bank mới đang được tạo. Phần bên dưới vẫn là artifact "
            "random-crop cũ để A/B và truy vết lỗi, chưa phải data rebuild."
        )
    report_path = args.audibility_report.resolve()
    if not report_path.is_file():
        st.error(f"Thiếu sound-data index: {report_path}")
        return
    report = _read_audibility_report(
        str(report_path), report_path.stat().st_mtime_ns
    )
    raw_events = report.get("events", [])
    manifest_identity = report.get("manifest", {})
    if not isinstance(raw_events, list) or not isinstance(manifest_identity, Mapping):
        st.error("Audibility report không có event index hợp lệ.")
        return
    manifest_text = manifest_identity.get("path")
    if not isinstance(manifest_text, str):
        st.error("Audibility report không bind manifest path.")
        return
    manifest_root = Path(manifest_text).resolve().parent
    indexed_events = [event for event in raw_events if isinstance(event, Mapping)]
    if not indexed_events:
        st.error("Sound-data index rỗng.")
        return

    class_rows = []
    labels = sorted(
        {str(event["label"]) for event in indexed_events if event.get("label")}
    )
    for label in labels:
        items = [event for event in indexed_events if event.get("label") == label]
        crop_keys = {
            (
                event.get("source_id"),
                tuple(event.get("source_crop_interval_seconds", [])),
            )
            for event in items
        }
        source_ids = {event.get("source_id") for event in items}
        class_rows.append(
            {
                "loại âm": humanize_event_label(label),
                "kind": items[0].get("event_kind"),
                "unique source ↑": len(source_ids),
                "unique crop ↑": len(crop_keys),
                "rendered occurrence ↑": len(items),
                "mean SNR dB ↑": round(
                    sum(float(item["event_to_background_snr_db ↑"]) for item in items)
                    / len(items),
                    2,
                ),
                "high-risk rate ↓": round(
                    sum(
                        item.get("screening_band")
                        == "high_masking_or_low_level_risk"
                        for item in items
                    )
                    / len(items),
                    3,
                ),
            }
        )
    with st.expander("Tổng quan tất cả loại âm", expanded=True):
        st.dataframe(class_rows, use_container_width=True, hide_index=True)

    controls = st.columns((1.2, 1.2, 1.0))
    with controls[0]:
        kind = st.selectbox(
            "Nhóm data",
            ("semantic", "nuisance", "tất cả"),
            key="sound_explorer_kind",
        )
    filtered_labels = [
        label
        for label in labels
        if kind == "tất cả"
        or any(
            event.get("label") == label and event.get("event_kind") == kind
            for event in indexed_events
        )
    ]
    with controls[1]:
        default_label = "Meow" if "Meow" in filtered_labels else filtered_labels[0]
        selected_label = st.selectbox(
            "Loại âm",
            filtered_labels,
            index=filtered_labels.index(default_label),
            format_func=humanize_event_label,
            key="sound_explorer_label",
        )
    label_events = [
        event for event in indexed_events if event.get("label") == selected_label
    ]
    split_options = ["tất cả", *sorted({str(event.get("split")) for event in label_events})]
    with controls[2]:
        selected_split = st.selectbox(
            "Split", split_options, key="sound_explorer_split"
        )
    if selected_split != "tất cả":
        label_events = [
            event for event in label_events if event.get("split") == selected_split
        ]

    unique_only = st.toggle(
        "Chỉ hiện unique source crop",
        value=True,
        help=(
            "Bỏ các bản render lặp lại cùng source/crop giữa base, order-swap "
            "và anchor-drop."
        ),
        key="sound_explorer_unique",
    )
    if unique_only:
        unique_events: list[Mapping[str, Any]] = []
        seen: set[tuple[object, ...]] = set()
        for event in label_events:
            key = (
                event.get("source_id"),
                tuple(event.get("source_crop_interval_seconds", [])),
                round(float(event.get("recipe_gain_db", 0.0)), 6),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_events.append(event)
        label_events = unique_events
    sort_mode = st.radio(
        "Sắp xếp sample",
        ("SNR thấp trước", "activity thấp trước", "scene / onset"),
        horizontal=True,
        key="sound_explorer_sort",
    )
    if sort_mode == "SNR thấp trước":
        label_events.sort(key=lambda event: float(event["event_to_background_snr_db ↑"]))
    elif sort_mode == "activity thấp trước":
        label_events.sort(key=lambda event: float(event["within_interval_activity_fraction ↑"]))
    else:
        label_events.sort(
            key=lambda event: (
                str(event.get("scene_id")), float(event.get("onset_seconds", 0.0))
            )
        )
    if not label_events:
        st.warning("Không có occurrence phù hợp filter.")
        return

    def event_label(event: Mapping[str, Any]) -> str:
        crop = event.get("source_crop_interval_seconds", [0.0, 0.0])
        return (
            f"source {event.get('source_id')} · crop "
            f"{float(crop[0]):.2f}–{float(crop[1]):.2f}s · "
            f"SNR {float(event['event_to_background_snr_db ↑']):+.2f} dB · "
            f"{event.get('scene_id')}:{event.get('event_id')}"
        )

    selected = st.selectbox(
        f"Occurrence của {humanize_event_label(selected_label)} "
        f"({len(label_events)} sample sau filter)",
        label_events,
        format_func=event_label,
        key="sound_explorer_occurrence",
    )
    crop = selected.get("source_crop_interval_seconds", [0.0, 0.0])
    onset = float(selected.get("onset_seconds", 0.0))
    offset = float(selected.get("offset_seconds", onset))
    st.dataframe(
        [
            {
                "metric": "Stem RMS dBFS ↑",
                "value": f"{float(selected['interval_stem_rms_dbfs ↑']):.2f}",
            },
            {
                "metric": "Stem peak dBFS ↑",
                "value": f"{float(selected['interval_stem_peak_dbfs ↑']):.2f}",
            },
            {
                "metric": "Event/background SNR dB ↑",
                "value": f"{float(selected['event_to_background_snr_db ↑']):.2f}",
            },
            {
                "metric": "Within-interval activity ↑",
                "value": f"{float(selected['within_interval_activity_fraction ↑']):.3f}",
            },
            {"metric": "Screening band", "value": selected.get("screening_band")},
        ],
        use_container_width=True,
        hide_index=True,
    )

    stem_path = _safe_child(manifest_root, selected.get("stem_path"))
    mixture_path = _safe_child(manifest_root, selected.get("mixture_path"))
    source_path = _safe_child(PROJECT_ROOT, selected.get("source_path"))
    players = st.columns(4)
    with players[0]:
        st.markdown("**1 · Source crop normalize để soi nội dung**")
        if source_path is None:
            st.error("Thiếu source audio.")
        else:
            payload, stats = _audio_excerpt(
                str(source_path),
                float(crop[0]),
                float(crop[1]),
                True,
                source_path.stat().st_mtime_ns,
            )
            st.audio(payload, format="audio/wav")
            st.caption(
                f"Inspection gain {stats['inspection_gain_db']:+.1f} dB; "
                "không phải level model nhận."
            )
    with players[1]:
        st.markdown("**2 · Rendered crop đúng level scene**")
        if stem_path is None:
            st.error("Thiếu rendered stem.")
        else:
            payload, stats = _audio_excerpt(
                str(stem_path), onset, offset, False, stem_path.stat().st_mtime_ns
            )
            st.audio(payload, format="audio/wav")
            st.caption(
                f"RMS {stats['original_rms_dbfs ↑']:.1f} dBFS ↑ · "
                f"peak {stats['original_peak_dbfs ↑']:.1f} dBFS ↑"
            )
    with players[2]:
        st.markdown("**3 · Full stem 10 giây đúng onset**")
        if stem_path is None:
            st.error("Thiếu rendered stem.")
        else:
            st.audio(str(stem_path), format="audio/wav")
            st.caption(f"Event nằm tại {onset:.2f}–{offset:.2f}s.")
    with players[3]:
        st.markdown("**4 · Mixture model thực sự nhận**")
        if mixture_path is None:
            st.error("Thiếu mixture.")
        else:
            st.audio(str(mixture_path), format="audio/wav")
            st.caption("Dùng player này để kết luận masking/audibility.")

    scene_id = selected.get("scene_id")
    scene_events = sorted(
        [event for event in indexed_events if event.get("scene_id") == scene_id],
        key=lambda event: float(event.get("onset_seconds", 0.0)),
    )
    st.markdown("#### Các âm cùng scene có thể che sample đang chọn")
    st.dataframe(
        [
            {
                "selected": "← đang nghe" if event.get("event_id") == selected.get("event_id") else "",
                "âm": f"{humanize_event_label(str(event.get('label')))} #{event.get('occurrence_index')}",
                "kind": event.get("event_kind"),
                "time": f"{float(event.get('onset_seconds', 0.0)):.2f}–{float(event.get('offset_seconds', 0.0)):.2f}s",
                "SNR dB ↑": round(float(event["event_to_background_snr_db ↑"]), 2),
                "screen": event.get("screening_band"),
            }
            for event in scene_events
        ],
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Metadata và source audio đầy đủ"):
        st.json(
            {
                "split": selected.get("split"),
                "scene_id": scene_id,
                "scene_family_id": selected.get("scene_family_id"),
                "variant_id": selected.get("variant_id"),
                "event_id": selected.get("event_id"),
                "event_kind": selected.get("event_kind"),
                "label": selected.get("label"),
                "source_id": selected.get("source_id"),
                "source_path": selected.get("source_path"),
                "source_interval_seconds": selected.get("source_interval_seconds"),
                "selected_source_crop_seconds": crop,
                "rendered_interval_seconds": [onset, offset],
                "recipe_gain_db": selected.get("recipe_gain_db"),
                "attribution": selected.get("attribution"),
            }
        )
        if source_path is not None:
            st.markdown("**Full source gốc, chưa crop**")
            st.audio(str(source_path), format="audio/wav")
            st.caption(
                f"Crop hiện tại nằm tại {float(crop[0]):.2f}–"
                f"{float(crop[1]):.2f}s trong source này."
            )


def render_realdesed_explorer(args: argparse.Namespace) -> None:
    st.subheader("RealDESED · audio thật và timestamp đã review")
    st.success(
        "Đây là ghi âm trong nhà thật, không phải scene synthetic. Mỗi player "
        "dùng đúng canonical crop 10 giây; event interval và QA evidence đều "
        "cùng một hệ thời gian 0–10 giây."
    )
    st.warning(
        "RealDESED không có clean source stem. Vì vậy crop event dưới đây vẫn "
        "có thể chứa âm chồng lấn; nó dùng để kiểm tra nhãn/timestamp và temporal "
        "grounding, không phải target waveform và không được tính SDR."
    )
    root = args.realdesed_root.resolve()
    scene_path = root / "scenes.scoring.jsonl"
    scoring_path = root / "qces_realdesed_scoring.jsonl"
    report_path = root / "build_report.json"
    missing = [path for path in (scene_path, scoring_path, report_path) if not path.is_file()]
    if missing:
        st.info(
            "RealDESED real-dev đang được build hoặc chưa có đủ artifact: "
            + ", ".join(str(path) for path in missing)
        )
        return
    try:
        scenes = _read_jsonl(
            str(scene_path), scene_path.stat().st_mtime_ns
        )
        scoring = _read_jsonl(
            str(scoring_path), scoring_path.stat().st_mtime_ns
        )
        report = _read_json(report_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Không đọc được RealDESED contract: {error}")
        return
    if not scenes:
        st.error("RealDESED scene manifest rỗng.")
        return

    counts = report.get("counts", {})
    summary = st.columns(4)
    summary[0].metric("Real scenes ↑", counts.get("eligible_scenes_↑", len(scenes)))
    summary[1].metric("Questions ↑", counts.get("questions_↑", len(scoring)))
    summary[2].metric(
        "Answerable ↑", counts.get("answerable_questions_↑", "?")
    )
    summary[3].metric(
        "Clean stems", "0 · SDR forbidden"
    )

    audit_root = args.realdesed_audit_root.resolve()
    queue_path = audit_root / "manual_listening_queue.jsonl"
    decision_path = audit_root / "manual_listening_decisions.json"
    queue_rows = (
        _read_jsonl(str(queue_path), queue_path.stat().st_mtime_ns)
        if queue_path.is_file()
        else []
    )
    decisions = _read_json(decision_path) if decision_path.is_file() else {}
    real_scope = st.radio(
        "Phạm vi RealDESED",
        ("30 case QC bắt buộc", "Toàn bộ 826 scene"),
        horizontal=True,
        disabled=not queue_rows,
        key="realdesed_scope",
    )
    real_queue_row: Mapping[str, Any] | None = None
    scene_index = {str(scene["scene_id"]): scene for scene in scenes}
    if real_scope == "30 case QC bắt buộc" and queue_rows:
        reviewed = sum(
            f"{row['scene_id']}::{row['event_id']}" in decisions
            for row in queue_rows
        )
        qc_columns = st.columns(4)
        qc_columns[0].metric("Đã nghe ↑", reviewed)
        qc_columns[1].metric("Còn lại ↓", len(queue_rows) - reviewed)
        qc_columns[2].metric(
            "Accept ↑",
            sum(
                decisions.get(f"{row['scene_id']}::{row['event_id']}", {}).get(
                    "decision"
                )
                == "accept"
                for row in queue_rows
            ),
        )
        qc_columns[3].metric(
            "Reject/uncertain ↓",
            sum(
                decisions.get(f"{row['scene_id']}::{row['event_id']}", {}).get(
                    "decision"
                )
                in {"reject", "uncertain"}
                for row in queue_rows
            ),
        )

        def real_queue_label(row: Mapping[str, Any]) -> str:
            key = f"{row['scene_id']}::{row['event_id']}"
            status = decisions.get(key, {}).get("decision", "CHƯA NGHE")
            interval = row["canonical_interval"]
            return (
                f"{row['display_name']} · {float(interval[0]):.2f}–"
                f"{float(interval[1]):.2f}s · {row['scene_id']} · {status}"
            )

        real_queue_row = st.selectbox(
            "Case RealDESED cần duyệt",
            queue_rows,
            format_func=real_queue_label,
            key="realdesed_queue_case",
        )
        selected_scene = scene_index[str(real_queue_row["scene_id"])]
    else:
        selected_scene = st.selectbox(
            "Chọn real scene",
            scenes,
            format_func=lambda scene: (
                f"{scene['scene_id']} · source {scene['upstream_record_id']} · "
                f"{len(scene.get('events', []))} reviewed events"
            ),
            key="realdesed_scene",
        )
    audio_path = _safe_child(root, selected_scene.get("canonical_audio_path"))
    if audio_path is None:
        st.error("Canonical audio bị thiếu hoặc thoát khỏi dataset root.")
        return
    st.markdown("### 1 · Nghe đúng real mixture 10 giây")
    st.audio(str(audio_path), format="audio/wav")
    metadata = selected_scene.get("metadata", {})
    st.caption(
        f"Upstream `{selected_scene['upstream_record_id']}` · crop absolute "
        f"{float(selected_scene['crop_start_seconds']):.3f}–"
        f"{float(selected_scene['crop_end_seconds']):.3f}s · "
        f"device `{metadata.get('recording_device', 'unknown')}` · "
        f"placement `{metadata.get('device_placement', 'unknown')}`"
    )
    with st.expander("Provenance, license và mô tả người thu"):
        st.json(
            {
                "dataset": selected_scene.get("dataset"),
                "Zenodo record": selected_scene.get("dataset_record_id"),
                "upstream split": selected_scene.get("upstream_split"),
                "upstream record": selected_scene.get("upstream_record_id"),
                "source SHA256": selected_scene.get("source_audio_sha256"),
                "canonical SHA256": selected_scene.get("canonical_audio_sha256"),
                "license/attribution exact": selected_scene.get("source_license"),
                "recording environment": metadata.get("recording_environment"),
                "scene description": metadata.get("scene_description"),
                "target classes supplied upstream": metadata.get("target_classes"),
                "non-target classes supplied upstream": metadata.get(
                    "non_target_classes"
                ),
            }
        )

    events = selected_scene.get("events", [])
    if not isinstance(events, list) or not events:
        st.error("Scene không có reviewed event sau crop.")
        return
    st.markdown("### 2 · Reviewed events nằm ở đâu?")
    st.dataframe(
        [
            {
                "event": event.get("event_id"),
                "sound": event.get("display_name"),
                "canonical time": _format_intervals(
                    [event.get("relative_interval")]
                ),
                "source annotation": _format_intervals(
                    [event.get("source_annotation_interval")]
                ),
                "right-censored at crop edge": event.get(
                    "right_censored_by_crop", False
                ),
                "interval RMS dBFS ↑": event.get(
                    "acoustic_diagnostic_not_semantic_proof", {}
                ).get("interval_rms_dbfs_↑"),
                "interval/context dB ↑": event.get(
                    "acoustic_diagnostic_not_semantic_proof", {}
                ).get("interval_to_context_rms_db_↑"),
            }
            for event in events
        ],
        use_container_width=True,
        hide_index=True,
    )
    event_default_index = 0
    if real_queue_row is not None:
        event_default_index = next(
            (
                index
                for index, event in enumerate(events)
                if event.get("event_id") == real_queue_row.get("event_id")
            ),
            0,
        )
    selected_event = st.selectbox(
        "Chọn event để nghe đúng interval",
        events,
        index=event_default_index,
        format_func=lambda event: (
            f"{event['display_name']} · "
            f"{float(event['relative_interval'][0]):.2f}–"
            f"{float(event['relative_interval'][1]):.2f}s"
        ),
        key=f"realdesed_event_{selected_scene['scene_id']}",
    )
    onset, offset = (float(value) for value in selected_event["relative_interval"])
    boosted, boosted_stats = _audio_excerpt(
        str(audio_path), onset, offset, True, audio_path.stat().st_mtime_ns
    )
    true_level, true_stats = _audio_excerpt(
        str(audio_path), onset, offset, False, audio_path.stat().st_mtime_ns
    )
    event_columns = st.columns(2)
    with event_columns[0]:
        st.markdown("**A · Mixture interval boost để nhận diện**")
        st.audio(boosted, format="audio/wav")
        st.caption(
            f"Canonical {onset:.2f}–{offset:.2f}s → player 0.00–"
            f"{offset - onset:.2f}s · boost "
            f"{boosted_stats['inspection_gain_db']:+.1f} dB."
        )
    with event_columns[1]:
        st.markdown("**B · Cùng interval, đúng level model nhận**")
        st.audio(true_level, format="audio/wav")
        st.caption(
            f"RMS {true_stats['original_rms_dbfs ↑']:.1f} dBFS ↑ · "
            f"peak {true_stats['original_peak_dbfs ↑']:.1f} dBFS ↑."
        )
    st.info(
        f"Cần nghe xem `{selected_event['display_name']}` có thực sự xuất hiện "
        f"trong {onset:.2f}–{offset:.2f}s. Nếu không chắc, note scene/event này "
        "để đưa vào manual rejection list; không tự đổi timestamp theo cảm giác."
    )
    if selected_event.get("right_censored_by_crop"):
        st.caption(
            "Event này còn tiếp tục sau mép phải của crop. Onset gốc vẫn nằm "
            "trong cửa sổ; offset chấm temporal được khai báo right-censored tại "
            "9.95s, không giả vờ đó là offset tự nhiên của source event."
        )
    if real_queue_row is not None:
        decision_key = (
            f"{selected_scene['scene_id']}::{selected_event['event_id']}"
        )
        previous = decisions.get(decision_key, {})
        decision_options = ("accept", "reject", "uncertain")
        previous_decision = previous.get("decision", "uncertain")
        decision = st.radio(
            "Kết luận case RealDESED",
            decision_options,
            index=(
                decision_options.index(previous_decision)
                if previous_decision in decision_options
                else 2
            ),
            horizontal=True,
            key=f"realdesed_decision_{decision_key}",
        )
        field_columns = st.columns(2)
        label_audible = field_columns[0].selectbox(
            "Label nghe nhận ra được?",
            ("yes", "no", "unsure"),
            index=("yes", "no", "unsure").index(
                previous.get("label_audible", "unsure")
                if previous.get("label_audible", "unsure")
                in {"yes", "no", "unsure"}
                else "unsure"
            ),
            key=f"realdesed_label_audible_{decision_key}",
        )
        timestamp_acceptable = field_columns[1].selectbox(
            "Timestamp chấp nhận được?",
            ("yes", "no", "unsure"),
            index=("yes", "no", "unsure").index(
                previous.get("timestamp_acceptable", "unsure")
                if previous.get("timestamp_acceptable", "unsure")
                in {"yes", "no", "unsure"}
                else "unsure"
            ),
            key=f"realdesed_timestamp_{decision_key}",
        )
        note = st.text_input(
            "Ghi chú QC RealDESED",
            value=str(previous.get("notes", "")),
            key=f"realdesed_note_{decision_key}",
        )
        if st.button(
            "Lưu kết luận RealDESED",
            type="primary",
            key=f"realdesed_save_{decision_key}",
        ):
            latest = _read_json(decision_path) if decision_path.is_file() else {}
            latest[decision_key] = {
                "decision": decision,
                "label_audible": label_audible,
                "timestamp_acceptable": timestamp_acceptable,
                "notes": note.strip(),
                "scene_id": str(selected_scene["scene_id"]),
                "event_id": str(selected_event["event_id"]),
                "class": str(selected_event["class"]),
                "canonical_interval": list(selected_event["relative_interval"]),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _json_write(decision_path, latest)
            st.success("Đã lưu kết luận RealDESED.")

    scene_questions = [
        row for row in scoring if row.get("scene_id") == selected_scene.get("scene_id")
    ]
    st.markdown("### 3 · Từ reviewed timeline sinh câu hỏi gì?")
    if not scene_questions:
        st.warning("Scene không có QA row.")
        return
    selected_question = st.selectbox(
        "Chọn QA",
        scene_questions,
        format_func=lambda row: f"{row['relation']} · {row['question']}",
        key=f"realdesed_question_{selected_scene['scene_id']}",
    )
    st.info(f"**Question:** {selected_question['question']}")
    st.success(f"**Gold answer (real-dev audit):** {selected_question['answer']}")
    if selected_question.get("no_evidence"):
        st.markdown(
            "Anchor class không xuất hiện trong reviewed event inventory → "
            "gold là `no_evidence`, temporal evidence rỗng."
        )
    else:
        role_columns = st.columns(2)
        role_columns[0].markdown(
            "**Anchor interval**\n\n"
            + _format_intervals(selected_question.get("anchor_intervals"))
        )
        role_columns[1].markdown(
            "**Answer interval**\n\n"
            + _format_intervals(selected_question.get("answer_intervals"))
        )
        st.caption(
            "Các interval này là temporal oracle trên mixture thật; không phải "
            "oracle clean waveform."
        )
    st.dataframe(
        [
            {
                "metric": "Temporal IoU ↑",
                "real-dev allowed": "yes",
            },
            {
                "metric": "QA sufficiency ↑ / residual leakage ↓",
                "real-dev allowed": "yes · model/human auditor",
            },
            {
                "metric": "SI-SDR / SD-SDR",
                "real-dev allowed": "NO · không có clean stem",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_health(health: Mapping[str, Any] | None) -> None:
    if health is None:
        return
    st.success(
        f"Checkpoint được phép chạy demo · profile `{health['profile']}` · tất cả gate PASS"
    )
    st.dataframe(
        [
            {
                "metric": gate["metric"] + " " + gate["direction"],
                "value": gate["value"],
                "gate": f"{gate['operator']} {gate['threshold']}",
                "pass": "PASS" if gate["passed"] else "FAIL",
            }
            for gate in health["gates"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_queue_status(path: Path) -> None:
    """Expose the durable experiment queue without granting inference permission."""

    if not path.is_file():
        return
    try:
        queue = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.warning(f"Không đọc được experiment queue: {error}")
        return
    if queue.get("format") != "qces_durable_gpu_queue_v1":
        st.warning(f"Experiment queue status sai format: {path}")
        return

    completed = {
        str(stage)
        for stage in queue.get("completed_stages", [])
        if isinstance(stage, str)
    }
    skipped_payload = queue.get("skipped_stages", {})
    skipped = skipped_payload if isinstance(skipped_payload, Mapping) else {}
    current = str(queue.get("current_stage", "unknown"))
    state = str(queue.get("state", "unknown"))
    with st.expander(
        "Tiến độ checkpoint thật", expanded=state != "completed_analysis_required"
    ):
        st.caption(
            "Queue chỉ chạy khi GPU đủ trống; app không tự dừng workload khác. "
            "Trạng thái này không thay thế checkpoint health gate."
        )
        status_columns = st.columns(3)
        status_columns[0].metric("Queue state", state)
        status_columns[1].metric("Current stage", current)
        status_columns[2].metric(
            "Completed stages ↑", f"{len(completed)} / {len(GPU_QUEUE_STAGES)}"
        )
        gpu = queue.get("last_gpu_sample")
        if isinstance(gpu, Mapping):
            gpu_columns = st.columns(3)
            gpu_columns[0].metric(
                "Queue GPU free ↑",
                f"{int(gpu['free_memory_mib_↑']):,} MiB",
            )
            gpu_columns[1].metric(
                "Queue GPU utilization ↓",
                f"{int(gpu['utilization_percent_↓'])}%",
            )
            gpu_columns[2].metric(
                "Queue GPU temperature ↓",
                f"{int(gpu['temperature_celsius_↓'])}°C",
            )
        st.dataframe(
            [
                {
                    "stage": stage,
                    "status": (
                        "PASS"
                        if stage in completed
                        else (
                            "SKIP"
                            if stage in skipped
                            else "RUNNING / WAITING" if stage == current else "PENDING"
                        )
                    ),
                    "detail": str(skipped.get(stage, "")),
                }
                for stage in GPU_QUEUE_STAGES
            ],
            use_container_width=True,
            hide_index=True,
        )
        if queue.get("error"):
            st.error(str(queue["error"]))
        st.caption(f"Updated UTC: {queue.get('updated_at_utc', 'unknown')}")


def _render_outputs(run_dir: Path) -> None:
    grounding_path = run_dir / "qces" / "grounding.json"
    if not grounding_path.is_file():
        return
    grounding = _read_json(grounding_path)
    request_path = run_dir / "request.json"
    request = _read_json(request_path) if request_path.is_file() else {}
    answer_path = run_dir / "audioqa.json"
    answer = _read_json(answer_path) if answer_path.is_file() else None

    st.divider()
    st.header("Kết quả theo từng bước")
    st.markdown("### 1 · Model nghe thấy audio có gì?")
    exact_window = run_dir / "qces" / "mixture.wav"
    if exact_window.is_file():
        st.audio(str(exact_window), format="audio/wav")
        st.caption("Đây là đúng cửa sổ 10 giây đã đưa vào QCES và AF3 caption.")
    inventory = answer.get("scene_inventory") if isinstance(answer, Mapping) else None
    if isinstance(inventory, Mapping) and inventory.get("status") == "valid":
        inventory_events = inventory.get("events", [])
        _render_event_inventory(
            inventory_events if isinstance(inventory_events, list) else [],
            predicted=True,
        )
    elif isinstance(inventory, Mapping):
        st.warning(
            "AF3 không trả được inventory JSON hợp lệ; không biến text lỗi thành "
            "nhãn âm thanh."
        )
        if inventory.get("raw_text"):
            st.caption(f"Raw AF3: {inventory['raw_text']}")
        if inventory.get("error"):
            st.caption(f"Caption error: {inventory['error']}")
    else:
        st.info("Run cũ chưa có bước AF3 scene inventory.")

    question = str(request.get("question", grounding.get("question", "")))
    relation = request.get("relation")
    st.markdown("### 2 · Câu hỏi yêu cầu gì?")
    st.info(f"**Câu hỏi:** {question}")
    st.markdown(f"**Yêu cầu:** {relation_requirement(relation, question)}")

    st.markdown("### 3 · QCES tìm evidence nào?")
    probability = float(grounding["no_evidence_probability"])
    roles = grounding.get("role_intervals", {})
    anchor_intervals = roles.get("anchor", []) if isinstance(roles, Mapping) else []
    answer_intervals = roles.get("answer", []) if isinstance(roles, Mapping) else []
    evidence_intervals = [*anchor_intervals, *answer_intervals]
    if probability >= 0.5:
        st.warning(
            f"QCES chọn **no evidence** · P(no evidence)={probability:.3f} "
            "(ngưỡng 0.5)."
        )
    elif evidence_intervals:
        role_columns = st.columns(2)
        role_columns[0].success(
            "**Anchor candidate**\n\n" + _format_intervals(anchor_intervals)
        )
        role_columns[1].success(
            "**Answer-event candidate**\n\n" + _format_intervals(answer_intervals)
        )
        st.caption(
            f"P(no evidence)={probability:.3f} (< 0.5). Tên âm ở đây chưa được "
            "gắn oracle; waveform và timestamps mới là acoustic rationale của QCES."
        )
    else:
        st.error(
            "QCES không có interval nào vượt ngưỡng 0.5 dù không chọn abstain; "
            "đây là dấu hiệu collapse/không nhất quán."
        )

    audio_columns = st.columns(3)
    output_audio = (
        ("Mixture X", exact_window),
        ("Evidence E · phần được giữ", run_dir / "qces" / "evidence.wav"),
        ("Residual R · phần bị bỏ", run_dir / "qces" / "residual.wav"),
    )
    for column, (label, path) in zip(audio_columns, output_audio):
        with column:
            st.markdown(f"**{label}**")
            if path.is_file():
                st.audio(str(path), format="audio/wav")

    if answer_path.is_file():
        assert answer is not None
        prediction = answer["prediction"]
        st.markdown("### 4 · Evidence dẫn đến câu trả lời nào?")
        st.success(
            f"AF3 chỉ nghe **Evidence E** và chọn: "
            f"**{prediction['label']}. {humanize_event_label(prediction['answer'])}** · "
            f"confidence ↑ {float(prediction['confidence_↑']):.3f}"
        )
        reference_path = run_dir / "benchmark_reference_after_inference.json"
        if reference_path.is_file():
            reference = _read_json(reference_path)
            correct = prediction["answer"] == reference["answer"]
            st.metric("Benchmark answer accuracy ↑", "1 / 1" if correct else "0 / 1")
            with st.expander("Đối chiếu oracle sau inference"):
                st.markdown(
                    f"**Gold answer:** {reference['answer_option_label']}. "
                    f"{humanize_event_label(reference['answer'])}"
                )
                if reference.get("no_evidence"):
                    st.markdown("**Gold evidence:** silence / no evidence")
                else:
                    reference_events = reference.get("scene_events", [])
                    st.markdown(
                        "**Gold anchor:** "
                        + _event_names(
                            reference_events, reference.get("anchor_event_ids", [])
                        )
                        + " · "
                        + _format_intervals(reference.get("anchor_intervals"))
                    )
                    st.markdown(
                        "**Gold answer event:** "
                        + _event_names(
                            reference_events, reference.get("answer_event_ids", [])
                        )
                        + " · "
                        + _format_intervals(reference.get("answer_intervals"))
                    )
                st.caption(
                    f"Oracle chỉ được mở sau inference · item `{reference['item_id']}`"
                )
    elif (run_dir / "audioqa_error.txt").is_file():
        st.error("QCES đã chạy xong nhưng Audio Flamingo thất bại.")
        st.code((run_dir / "audioqa_error.txt").read_text(encoding="utf-8"))

    with st.expander("Thông số và output kỹ thuật"):
        metrics = st.columns(3)
        metrics[0].metric("P(no evidence)", f"{probability:.3f}")
        metrics[1].metric(
            "Mixture consistency L1 ↓",
            f"{float(grounding['mixture_consistency_l1']):.3e}",
        )
        metrics[2].metric(
            "Window", f"{float(grounding['duration_seconds']):.1f}s"
        )
        st.dataframe(
            [
                {"role": role, "predicted intervals": _format_intervals(value)}
                for role, value in roles.items()
            ]
            if isinstance(roles, Mapping)
            else [],
            use_container_width=True,
            hide_index=True,
        )
        if isinstance(answer, Mapping):
            st.dataframe(answer["options"], use_container_width=True, hide_index=True)
        st.caption(f"Run artifacts: {run_dir}")


def render_live(args: argparse.Namespace) -> None:
    st.subheader("Live inference · Audio + Question → Evidence + Answer")
    _render_queue_status(args.queue_status)
    assets_ready, errors, health = _asset_readiness(args)
    gpu = _gpu_state()
    resource_ready = gpu_resource_ready(
        gpu,
        args.minimum_free_gpu_mib,
        args.maximum_gpu_utilization_percent,
    )
    workflow_ready = exclusive_lock_available(args.gpu_workflow_lock.resolve())
    free_mib = gpu["free_memory_mib_↑"] if gpu is not None else None
    utilization = gpu["utilization_percent_↓"] if gpu is not None else None
    temperature = gpu["temperature_celsius_↓"] if gpu is not None else None
    status_columns = st.columns(5)
    status_columns[0].metric("Checkpoint gate ↑", "PASS" if assets_ready else "BLOCKED")
    status_columns[1].metric(
        "GPU free ↑", "unknown" if free_mib is None else f"{free_mib:,} MiB"
    )
    status_columns[2].metric(
        "GPU utilization ↓",
        "unknown" if utilization is None else f"{utilization}%",
    )
    status_columns[3].metric(
        "GPU temperature ↓",
        "unknown" if temperature is None else f"{temperature}°C",
    )
    status_columns[4].metric(
        "GPU workflow lock ↑", "FREE" if workflow_ready else "BUSY"
    )
    _render_health(health)
    for error in errors:
        st.error(error)
    if gpu is None:
        st.warning("Không đọc được trạng thái GPU; live inference bị khóa fail-closed.")
    elif free_mib < args.minimum_free_gpu_mib:
        st.warning(
            f"Cần ít nhất {args.minimum_free_gpu_mib:,} MiB GPU trống; hiện chỉ có "
            f"{free_mib:,} MiB. App không tự dừng process khác của m."
        )
    elif utilization > args.maximum_gpu_utilization_percent:
        st.warning(
            f"GPU đang bận {utilization}%; cần ≤ "
            f"{args.maximum_gpu_utilization_percent}% trước khi inference."
        )
    if not workflow_ready:
        st.warning(
            "Experiment queue đang giữ global GPU lock; live inference đợi queue "
            "kết thúc để không tranh VRAM."
        )

    st.info(
        "Model dùng đúng một cửa sổ 10 giây. Audio ngắn hơn sẽ được pad silence bên "
        "phải; audio dài hơn dùng cửa sổ bắt đầu tại thời điểm m chọn."
    )
    examples: list[dict[str, Any]] = []
    if health is not None:
        try:
            examples = load_fingerprint_bound_examples(health)
        except DemoContractError as error:
            st.warning(f"Không mở được benchmark preset đã fingerprint: {error}")

    input_modes = ["Upload audio của m"]
    if examples:
        input_modes.insert(0, "Frozen listening case")
    input_mode = st.radio(
        "Demo input",
        input_modes,
        horizontal=True,
        help=(
            "Frozen cases được chọn trước khi thấy output và bind vào đúng manifest "
            "đã dùng để cấp health receipt."
        ),
    )
    benchmark_reference: dict[str, Any] | None = None
    if input_mode == "Frozen listening case":
        selected_example = st.selectbox(
            "Question case",
            examples,
            format_func=lambda item: (
                f"{item['relation']} | "
                f"{'no-evidence' if item['no_evidence'] else 'answerable'} | "
                f"{item['id']}"
            ),
        )
        st.markdown("### 1 · Nghe mixture và xem timeline")
        st.audio(selected_example["mixture_path"], format="audio/wav")
        _render_event_inventory(selected_example.get("scene_events", []), predicted=False)
        st.markdown("### 2 · Đọc yêu cầu")
        st.info(f"**Câu hỏi:** {selected_example['question']}")
        st.markdown(
            "**Model cần làm:** "
            + relation_requirement(
                selected_example.get("relation"), selected_example["question"]
            )
        )
        st.markdown("**Các đáp án có thể chọn:**")
        st.dataframe(
            [
                {"label": chr(65 + index), "option": option}
                for index, option in enumerate(selected_example["answer_options"])
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Đây là memorization/anti-collapse case nếu health profile là "
            "micro_overfit; không phải kết quả held-out. Gold chỉ hiện sau khi "
            "hai model đã inference xong."
        )
        with st.form("live_qces_frozen_request"):
            submitted = st.form_submit_button(
                "Run frozen case · QCES + Audio Flamingo",
                disabled=not (assets_ready and resource_ready and workflow_ready),
                use_container_width=True,
            )
        upload = None
        window_start = 0.0
        question = selected_example["question"]
        relation = selected_example.get("relation")
        options = list(selected_example["answer_options"])
        input_source = Path(selected_example["mixture_path"])
        benchmark_reference = {
            "format": "qces_streamlit_post_inference_reference_v1",
            "item_id": selected_example["id"],
            "answer": selected_example["answer"],
            "answer_option_index": selected_example["answer_option_index"],
            "answer_option_label": chr(65 + selected_example["answer_option_index"]),
            "relation": relation,
            "no_evidence": selected_example["no_evidence"],
            "scene_events": selected_example.get("scene_events", []),
            "anchor_event_ids": selected_example.get("anchor_event_ids", []),
            "answer_event_ids": selected_example.get("answer_event_ids", []),
            "anchor_intervals": selected_example.get("anchor_intervals", []),
            "answer_intervals": selected_example.get("answer_intervals", []),
            "written_after_inference": True,
        }
    else:
        with st.form("live_qces_upload_request"):
            upload = st.file_uploader("Audio", type=("wav", "flac", "ogg"))
            window_start = st.number_input(
                "Window start (seconds)", min_value=0.0, value=0.0, step=0.5
            )
            question = st.text_input("Question")
            option_count = st.selectbox(
                "Number of answer options", (2, 3, 4, 5), index=3
            )
            default_options = ("", "", "", "", "no_evidence")
            options = [
                st.text_input(f"{chr(65 + index)}", value=default_options[index])
                for index in range(option_count)
            ]
            submitted = st.form_submit_button(
                "Run QCES + Audio Flamingo",
                disabled=not (assets_ready and resource_ready and workflow_ready),
                use_container_width=True,
            )
        input_source = None
        relation = None

    if submitted:
        cleaned_question = question.strip()
        cleaned_options = [option.strip() for option in options]
        if input_source is None and upload is None:
            st.error("Chọn một audio file trước.")
            return
        if not cleaned_question:
            st.error("Question không được rỗng.")
            return
        if any(not option for option in cleaned_options):
            st.error("Mọi option phải có nội dung.")
            return
        if len({option.casefold() for option in cleaned_options}) != len(
            cleaned_options
        ):
            st.error("Các option phải khác nhau.")
            return

        if input_source is not None:
            payload = input_source.read_bytes()
            original_filename = input_source.name
            suffix = input_source.suffix.lower()
            benchmark_item_id = benchmark_reference["item_id"]
        else:
            payload = upload.getvalue()
            original_filename = upload.name
            suffix = Path(upload.name).suffix.lower()
            benchmark_item_id = None
        if suffix not in {".wav", ".flac", ".ogg"}:
            st.error("Định dạng audio không được hỗ trợ.")
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.runs_root.resolve() / f"{timestamp}_{uuid.uuid4().hex[:10]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        input_path = run_dir / f"input{suffix}"
        input_path.write_bytes(payload)
        _json_write(
            run_dir / "request.json",
            {
                "format": "qces_streamlit_request_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "question": cleaned_question,
                "relation": relation,
                "options": cleaned_options,
                "window_start_seconds": float(window_start),
                "input_filename": original_filename,
                "input_sha256": _sha256_bytes(payload),
                "input_mode": (
                    "fingerprint_bound_benchmark"
                    if benchmark_item_id is not None
                    else "user_upload"
                ),
                "benchmark_item_id": benchmark_item_id,
                "contains_gold_answer_input": False,
            },
        )
        qces_dir = run_dir / "qces"
        qces_command = [
            str(args.qces_python),
            str(CODE_ROOT / "mixi_understanding/scripts/infer_qces.py"),
            "--checkpoint",
            str(args.qces_checkpoint.resolve()),
            "--audiosep-root",
            str(args.audiosep_root.resolve()),
            "--audiosep-config",
            str(args.audiosep_config.resolve()),
            "--audiosep-checkpoint",
            str(args.audiosep_checkpoint.resolve()),
            "--audio",
            str(input_path),
            "--question",
            cleaned_question,
            "--window-start-seconds",
            str(float(window_start)),
            "--output-dir",
            str(qces_dir),
            "--device",
            "cuda",
        ]
        try:
            with _exclusive_demo_lock(args.gpu_workflow_lock.resolve()):
                with _exclusive_demo_lock(args.runs_root.resolve() / ".inference.lock"):
                    with st.spinner("QCES đang tách evidence/residual…"):
                        qces_result = _run(qces_command, timeout_seconds=900)
                    (run_dir / "qces_stdout.txt").write_text(
                        qces_result.stdout, encoding="utf-8"
                    )
                    (run_dir / "qces_stderr.txt").write_text(
                        qces_result.stderr, encoding="utf-8"
                    )
                    if qces_result.returncode != 0:
                        st.error("QCES inference thất bại.")
                        st.code(_tail(qces_result.stderr or qces_result.stdout))
                        st.session_state["qces_last_run"] = str(run_dir)
                        return

                    audioqa_command = [
                        str(args.audioqa_python),
                        str(
                            CODE_ROOT
                            / "mixi_understanding/scripts/infer_qces_audioqa.py"
                        ),
                        "--audio",
                        str(qces_dir / "evidence.wav"),
                        "--mixture-audio",
                        str(qces_dir / "mixture.wav"),
                        "--question",
                        cleaned_question,
                        "--output",
                        str(run_dir / "audioqa.json"),
                        "--model",
                        str(args.af3_model.resolve()),
                        "--quantization",
                        "4bit",
                        "--dtype",
                        "float16",
                        "--device",
                        "auto",
                        "--device-map",
                        "auto",
                        "--attention-implementation",
                        "sdpa",
                        "--local-files-only",
                    ]
                    for option in cleaned_options:
                        audioqa_command.extend(("--option", option))
                    with st.spinner(
                        "Audio Flamingo 3 đang mô tả mixture rồi chấm đáp án "
                        "trên evidence…"
                    ):
                        audioqa_result = _run(audioqa_command, timeout_seconds=1_200)
                    (run_dir / "audioqa_stdout.txt").write_text(
                        audioqa_result.stdout, encoding="utf-8"
                    )
                    (run_dir / "audioqa_stderr.txt").write_text(
                        audioqa_result.stderr, encoding="utf-8"
                    )
                    if audioqa_result.returncode != 0:
                        (run_dir / "audioqa_error.txt").write_text(
                            _tail(audioqa_result.stderr or audioqa_result.stdout),
                            encoding="utf-8",
                        )
                    if benchmark_reference is not None:
                        _json_write(
                            run_dir / "benchmark_reference_after_inference.json",
                            benchmark_reference,
                        )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            st.error(f"Pipeline bị dừng: {error}")
        st.session_state["qces_last_run"] = str(run_dir)

    last_run = st.session_state.get("qces_last_run")
    if isinstance(last_run, str):
        run_dir = Path(last_run)
        if run_dir.is_dir():
            _render_outputs(run_dir)


def render_simple_demo(args: argparse.Namespace) -> None:
    """One-page frozen-case listener for the anti-collapse checkpoint."""

    st.title("QCES · Nghe bằng chứng theo câu hỏi")
    st.caption(
        "Chọn một case có sẵn → đọc câu hỏi → nghe X, E, E* và R. "
        "Mục tiêu là kiểm tra model giữ đúng âm cần để trả lời."
    )

    try:
        health = load_and_validate_health_receipt(
            args.qces_checkpoint.resolve(), args.health_receipt.resolve()
        )
        examples = load_fingerprint_bound_examples(health, maximum_examples=4)
    except (DemoContractError, OSError) as error:
        st.error(f"Không mở được checkpoint/evaluation packet: {error}")
        return

    report_path = Path(health["evaluation_report"]["path"]).resolve()
    report = _read_json(report_path)
    item_by_id = {
        str(item["id"]): item
        for item in report.get("items", [])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    options = [
        (
            f"{example['relation']} · "
            f"{'no-evidence' if example['no_evidence'] else 'answerable'} · "
            f"{example['id']}"
        )
        for example in examples
    ]
    selected_index = st.selectbox(
        "Chọn case để nghe",
        range(len(examples)),
        format_func=lambda index: options[index],
    )
    selected = examples[selected_index]
    evaluated = item_by_id.get(selected["id"], {})

    st.markdown("## 1 · Case này nói về gì?")
    st.info(f"**Câu hỏi:** {selected['question']}")
    st.markdown(
        "**Nói đơn giản:** "
        + relation_requirement(selected.get("relation"), selected["question"])
    )
    if selected["no_evidence"]:
        st.warning(
            "Đây là case **no-evidence**: câu hỏi không có bằng chứng âm thanh "
            "hợp lệ trong scene, nên evidence đúng phải gần như im lặng."
        )
    else:
        anchor = _event_names(selected["scene_events"], selected["anchor_event_ids"])
        answer_event = _event_names(
            selected["scene_events"], selected["answer_event_ids"]
        )
        st.markdown(
            f"**Âm mốc (anchor):** {anchor or 'xem timeline'}  \n"
            f"**Âm trả lời cần giữ:** {answer_event or 'xem target sau khi nghe'}"
        )
    with st.expander("Scene có những âm nào? (timeline recipe)", expanded=True):
        _render_event_inventory(selected.get("scene_events", []), predicted=False)

    st.markdown("## 2 · Nghe theo đúng thứ tự")
    st.caption(
        "X = audio gốc · E = evidence model dự đoán · E* = evidence target · "
        "R = residual model. Hãy so E với E*, không so với audio gốc bằng cảm giác âm lượng."
    )
    scene_dir = args.microfit_preview_root / str(evaluated.get("scene_id"))
    question_dir = scene_dir / (
        f"q{evaluated.get('question_index')}_{evaluated.get('question_type')}"
    )
    audio_paths = [
        ("X · Mixture gốc", scene_dir / "mixture.wav"),
        ("E · Predicted evidence", question_dir / "predicted_evidence.wav"),
        ("E* · Target evidence", question_dir / "target_evidence.wav"),
        ("R · Predicted residual", question_dir / "predicted_residual.wav"),
        ("R* · Target residual", question_dir / "target_residual.wav"),
    ]
    for row_start in range(0, len(audio_paths), 3):
        columns = st.columns(3)
        for column, (label, path) in zip(columns, audio_paths[row_start : row_start + 3]):
            with column:
                st.markdown(f"**{label}**")
                if path.is_file():
                    st.audio(str(path), format="audio/wav")
                else:
                    st.error(f"Thiếu file: {path.name}")

    st.markdown("## 3 · Model tìm đúng đến đâu?")
    st.dataframe(
        [
            {
                "metric": "Temporal IoU ↑",
                "value": evaluated.get("temporal_iou"),
                "meaning": "E có trùng đúng vùng target không",
            },
            {
                "metric": "Evidence SD-SDRi ↑",
                "value": evaluated.get("evidence_sd_sdri"),
                "meaning": "E sạch hơn mixture bao nhiêu",
            },
            {
                "metric": "P(no evidence)",
                "value": evaluated.get("no_evidence_probability"),
                "meaning": "model có nghĩ case này không có bằng chứng không",
            },
            {
                "metric": "Retained ratio ↓",
                "value": evaluated.get("retained_ratio"),
                "meaning": "tỷ lệ năng lượng được giữ trong E",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        f"**Model interval:** {_format_intervals(evaluated.get('predicted_evidence_intervals'))}  \n"
        f"**Target interval:** {_format_intervals(evaluated.get('target_evidence_intervals'))}"
    )
    with st.expander("Gold để đối chiếu sau khi đã nghe"):
        if selected["no_evidence"]:
            st.write("Gold: no-evidence / silence")
        else:
            st.write(
                f"Gold answer option: **{selected['answer']}** · "
                f"anchor interval: {_format_intervals(selected.get('anchor_intervals'))} · "
                f"answer interval: {_format_intervals(selected.get('answer_intervals'))}"
            )
    st.caption(
        "Đây là checkpoint micro-overfit 30 mẫu để kiểm tra chống collapse, "
        "không phải kết quả held-out hay kết quả paper."
    )


def main() -> None:
    args = parse_args(sys.argv[1:])
    st.set_page_config(
        page_title="QCES · Question-Conditioned Evidence Separation",
        page_icon="🎧",
        layout="wide",
    )
    render_simple_demo(args)


if __name__ == "__main__":
    main()
