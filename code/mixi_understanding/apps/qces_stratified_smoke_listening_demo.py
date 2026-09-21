#!/usr/bin/env python3
"""Single-page listening audit for the 24-item stratified AudioSep smoke v2."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# Load the pure-Python contract without importing ``mixi_understanding.qces``.
# The latter imports torch in its package initializer, while the lightweight
# Streamlit environment intentionally does not install torch.
CONTRACT_PATH = CODE_ROOT / "mixi_understanding/qces/stratified_smoke_listening.py"
CONTRACT_MODULE_NAME = "qces_stratified_smoke_listening_contract"
_contract_spec = importlib.util.spec_from_file_location(
    CONTRACT_MODULE_NAME, CONTRACT_PATH
)
if _contract_spec is None or _contract_spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load listening contract: {CONTRACT_PATH}")
_contract = importlib.util.module_from_spec(_contract_spec)
sys.modules[CONTRACT_MODULE_NAME] = _contract
_contract_spec.loader.exec_module(_contract)

SmokeListeningContractError = _contract.SmokeListeningContractError
filter_smoke_records = _contract.filter_smoke_records
format_interval = _contract.format_interval
load_smoke_listening_bundle = _contract.load_smoke_listening_bundle
metric_table_rows = _contract.metric_table_rows
not_gold_details = _contract.not_gold_details
rejection_details = _contract.rejection_details
resolve_artifact_path = _contract.resolve_artifact_path


DEFAULT_ROOT = PROJECT_ROOT / "outputs/qces_stratified_audiosep_smoke_v2"
DEFAULT_AUDIT = DEFAULT_ROOT / "audiosep_clean/quality_audit.jsonl"
DEFAULT_RECEIPT = DEFAULT_ROOT / "audiosep_clean/source_bank_receipt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def load_bundle(audit_manifest: str, receipt_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = load_smoke_listening_bundle(
        Path(audit_manifest),
        Path(receipt_path),
        project_root=PROJECT_ROOT,
        require_artifacts=True,
    )
    return list(bundle.records), bundle.receipt


@st.cache_data(show_spinner=False)
def read_audio(path: str) -> bytes:
    return Path(path).read_bytes()


def display_label(value: str) -> str:
    return value.replace("_and_", " / ").replace("_or_", " / ").replace("_", " ")


def _number_text(value: Any) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    return f"{number:.4f}"


def render_condition(detail: Mapping[str, Any]) -> None:
    code = str(detail["code"])
    value = _number_text(detail.get("value"))
    threshold = _number_text(detail.get("threshold"))
    operator = str(detail.get("operator", ""))
    gate = str(detail.get("gate", ""))
    st.markdown(
        f"- **`{code}`** · `{value}` cần `{operator} {threshold}` "
        f"ở gate `{gate}` — {detail['explanation']}"
    )


def render_item(
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    index: int,
    total: int,
    expand: bool,
) -> None:
    tier = str(row["acceptance_tier"])
    icon = {"rejected": "🔴", "silver": "🟡", "gold": "🟢"}[tier]
    label = display_label(str(row["canonical_display_name"]))
    title = f"{icon} {index:02d}/{total:02d} · {label} · {tier.upper()}"
    with st.expander(title, expanded=expand):
        metadata, timing = st.columns([1.15, 1])
        with metadata:
            st.markdown(f"**Nhãn cần tách:** `{row['label']}`")
            st.markdown(f"**Text prompt AudioSep:** `{row['canonical_prompt']}`")
            st.markdown(f"**Source route:** `{row['materialization_source_route']}`")
            st.markdown(
                f"**Split:** metadata `{row['metadata_split']}` · source bank `{row['source_split']}`"
            )
            st.markdown(f"**AudioSet video:** `{row['source_video_id']}`")
            st.caption(
                f"Upstream ambiguity: tier {row.get('ambiguity_tier', 'N/A')} · "
                f"{row.get('ambiguity_tier_name', 'unknown')}"
            )
        with timing:
            st.markdown(
                "**Crop trong clip AudioSet gốc:** "
                + format_interval(
                    row["source_crop_start_seconds"], row["source_crop_end_seconds"]
                )
            )
            st.markdown(
                "**Annotation trên player bên dưới:** "
                + format_interval(
                    row["strong_annotation_onset_seconds"],
                    row["strong_annotation_offset_seconds"],
                )
            )
            st.markdown(
                "**Audible support đo được:** "
                + format_interval(row["active_onset_seconds"], row["active_offset_seconds"])
            )
            st.markdown(f"**Độ dài player:** `{float(row['duration_seconds']):.3f} s`")
            st.caption(
                "Player bắt đầu tại 0 s vì đây đã là crop. Hãy nghe đúng cửa sổ annotation "
                "để xác nhận nhãn có thực sự rõ và đúng thời gian."
            )

        source_path = resolve_artifact_path(PROJECT_ROOT, str(row["source_audio_path"]))
        stem_value = str(row.get("stem_path", ""))
        if row["accepted"] and stem_value:
            source_column, stem_column = st.columns(2)
            with source_column:
                st.markdown("**1. Source crop trước khi tách**")
                st.audio(read_audio(str(source_path)), format="audio/flac")
            with stem_column:
                st.markdown(f"**2. AudioSep stem đã nhận · {tier.upper()}**")
                stem_path = resolve_artifact_path(PROJECT_ROOT, stem_value)
                st.audio(read_audio(str(stem_path)), format="audio/flac")
        else:
            st.markdown("**Source crop cần nghe kiểm tra**")
            st.audio(read_audio(str(source_path)), format="audio/flac")
            st.warning(
                "Mẫu bị loại nên cleaner không lưu candidate stem vào source bank. "
                "Đây là chủ ý của data contract, không phải player bị thiếu file."
            )

        if tier == "rejected":
            st.error("REJECTED — không được dùng để train; cần lấy mẫu reserve thay thế.")
            details = rejection_details(row, receipt)
            st.markdown("**Điều kiện làm mẫu bị loại:**")
            if details:
                for detail in details:
                    render_condition(detail)
            else:
                st.markdown("- Receipt không ghi được lý do loại; đây là lỗi contract cần kiểm tra.")
        elif tier == "silver":
            st.warning("SILVER — đủ ngưỡng tối thiểu và được nhận, nhưng chưa đạt Gold.")
            details = not_gold_details(row, receipt)
            if details:
                st.markdown("**Khoảng cách còn thiếu so với Gold:**")
                for detail in details:
                    render_condition(detail)
        else:
            st.success("GOLD — đạt toàn bộ ngưỡng chất lượng Gold.")

        with st.expander("Xem 8 quality metrics và ngưỡng", expanded=tier == "rejected"):
            st.dataframe(
                metric_table_rows(row, receipt),
                hide_index=True,
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="QCES · AudioSep smoke listening audit",
        page_icon="🎧",
        layout="wide",
    )
    args = parse_args()
    st.title("QCES · Audit 24 AudioSep smoke crops")
    st.caption(
        "Trang này chỉ để audit data smoke: nghe source crop, so với stem đã được nhận, "
        "và xác định chính xác vì sao một crop bị loại. Đây không phải benchmark model."
    )

    try:
        records, receipt = load_bundle(str(args.audit_manifest), str(args.receipt))
    except SmokeListeningContractError as exc:
        st.error(f"Data contract không hợp lệ: {exc}")
        st.stop()

    counts = {
        tier: sum(row["acceptance_tier"] == tier for row in records)
        for tier in ("gold", "silver", "rejected")
    }
    accepted = counts["gold"] + counts["silver"]
    total = len(records)
    summary_columns = st.columns(5)
    summary_columns[0].metric("Tổng crop", total)
    summary_columns[1].metric("Accepted ↑", f"{accepted}/{total}", f"{accepted / total:.1%}")
    summary_columns[2].metric("Gold ↑", counts["gold"])
    summary_columns[3].metric("Silver", counts["silver"])
    summary_columns[4].metric("Rejected ↓", counts["rejected"])

    with st.expander("Cách đọc thời gian và quality metrics", expanded=True):
        st.markdown(
            "- **Crop trong clip gốc**: vị trí lấy audio từ clip AudioSet khoảng 10 giây.\n"
            "- **Annotation trên player**: vị trí event tương đối trong source crop đang phát; "
            "đây là đoạn cần nghe.\n"
            "- **Audible support**: vùng model đo là còn nghe thấy trong stem; không thay thế "
            "annotation gốc.\n"
            "- `↑` càng cao thường càng tốt, `↓` càng thấp càng tốt, `↔` phải nằm trong khoảng.\n"
            "- **Gold** đạt ngưỡng chặt. **Silver** đạt ngưỡng tối thiểu và chỉ được nhận "
            "khi upstream ambiguity tier ≤ 1. **Rejected** không đi vào source bank."
        )

    st.subheader("Bộ lọc")
    filter_columns = st.columns([1.05, 1.6, 0.9, 1.1])
    with filter_columns[0]:
        selected_tiers = st.multiselect(
            "Acceptance tier",
            options=["rejected", "silver", "gold"],
            default=["rejected"],
            help="Mặc định chỉ hiện 6 mẫu bị loại cần nghe trước.",
        )
    routes = sorted({str(row["materialization_source_route"]) for row in records})
    with filter_columns[1]:
        selected_routes = st.multiselect("Source route", options=routes, default=[])
    splits = sorted({str(row["metadata_split"]) for row in records})
    with filter_columns[2]:
        selected_splits = st.multiselect("Metadata split", options=splits, default=[])
    with filter_columns[3]:
        query = st.text_input("Tìm label", value="", placeholder="Ví dụ: Slam")

    filtered = filter_smoke_records(
        records,
        tiers=selected_tiers,
        routes=selected_routes,
        metadata_splits=selected_splits,
        label_query=query,
    )
    st.subheader(f"Danh sách nghe · {len(filtered)}/{total} crop")
    if not filtered:
        st.info("Không có crop nào khớp bộ lọc.")
        return
    expand = len(filtered) <= counts["rejected"]
    for index, row in enumerate(filtered, start=1):
        render_item(row, receipt, index=index, total=len(filtered), expand=expand)


if __name__ == "__main__":
    main()
