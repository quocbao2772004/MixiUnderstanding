"""Data contract for the stratified AudioSep smoke listening audit.

The Streamlit app is deliberately kept separate from this module so the
contract can be tested without importing Streamlit.  Paths in the manifests
are project-relative provenance paths; callers must supply the project root
used to resolve them.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class SmokeListeningContractError(ValueError):
    """Raised when a smoke receipt or audit row violates its contract."""


REQUIRED_FIELDS = frozenset(
    {
        "acceptance_tier",
        "accepted",
        "active_offset_seconds",
        "active_onset_seconds",
        "canonical_display_name",
        "canonical_prompt",
        "duration_seconds",
        "item_id",
        "label",
        "materialization_source_route",
        "metadata_split",
        "quality_gate",
        "quality_metrics",
        "source_audio_path",
        "source_crop_end_seconds",
        "source_crop_start_seconds",
        "source_split",
        "source_video_id",
        "stem_path",
        "strong_annotation_offset_seconds",
        "strong_annotation_onset_seconds",
    }
)

TIER_ORDER = {"rejected": 0, "silver": 1, "gold": 2}


@dataclass(frozen=True)
class SmokeListeningBundle:
    """Validated audit rows and their aggregate receipt."""

    records: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]
    project_root: Path

    @property
    def acceptance_counts(self) -> dict[str, int]:
        return {
            tier: sum(row["acceptance_tier"] == tier for row in self.records)
            for tier in ("gold", "silver", "rejected")
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeListeningContractError(f"cannot read JSON receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SmokeListeningContractError(f"receipt must be a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise SmokeListeningContractError(
                        f"audit row {line_number} must be a JSON object"
                    )
                rows.append(value)
    except SmokeListeningContractError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeListeningContractError(f"cannot read JSONL audit {path}: {exc}") from exc
    return rows


def resolve_artifact_path(project_root: Path, value: str | Path) -> Path:
    """Resolve an absolute or project-relative manifest artifact path."""

    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SmokeListeningContractError(
            f"item {row.get('item_id', '<unknown>')} has non-numeric {key}"
        )
    return float(value)


def _validate_row(
    row: Mapping[str, Any], *, project_root: Path, require_artifacts: bool
) -> None:
    missing = sorted(REQUIRED_FIELDS.difference(row))
    if missing:
        raise SmokeListeningContractError(
            f"item {row.get('item_id', '<unknown>')} missing fields: {missing}"
        )

    tier = str(row["acceptance_tier"])
    if tier not in TIER_ORDER:
        raise SmokeListeningContractError(f"unknown acceptance tier: {tier}")
    accepted = bool(row["accepted"])
    if accepted != (tier in {"gold", "silver"}):
        raise SmokeListeningContractError(
            f"item {row['item_id']} has inconsistent accepted/tier fields"
        )

    duration = _number(row, "duration_seconds")
    crop_on = _number(row, "source_crop_start_seconds")
    crop_off = _number(row, "source_crop_end_seconds")
    ann_on = _number(row, "strong_annotation_onset_seconds")
    ann_off = _number(row, "strong_annotation_offset_seconds")
    active_on = _number(row, "active_onset_seconds")
    active_off = _number(row, "active_offset_seconds")
    tolerance = 2e-3
    if duration <= 0 or crop_on < 0 or crop_off <= crop_on:
        raise SmokeListeningContractError(f"item {row['item_id']} has invalid crop interval")
    if abs((crop_off - crop_on) - duration) > tolerance:
        raise SmokeListeningContractError(
            f"item {row['item_id']} crop duration disagrees with duration_seconds"
        )
    for name, onset, offset in (
        ("annotation", ann_on, ann_off),
        ("audible support", active_on, active_off),
    ):
        if onset < -tolerance or offset < onset or offset > duration + tolerance:
            raise SmokeListeningContractError(
                f"item {row['item_id']} has invalid {name} interval"
            )

    if not isinstance(row["quality_gate"], dict) or not isinstance(
        row["quality_metrics"], dict
    ):
        raise SmokeListeningContractError(
            f"item {row['item_id']} quality fields must be objects"
        )

    source_value = str(row["source_audio_path"])
    if not source_value:
        raise SmokeListeningContractError(f"item {row['item_id']} has no source audio path")
    stem_value = str(row["stem_path"])
    if accepted and not stem_value:
        raise SmokeListeningContractError(f"accepted item {row['item_id']} has no stem path")
    if require_artifacts:
        source = resolve_artifact_path(project_root, source_value)
        if not source.is_file():
            raise SmokeListeningContractError(f"source audio does not exist: {source}")
        if accepted:
            stem = resolve_artifact_path(project_root, stem_value)
            if not stem.is_file():
                raise SmokeListeningContractError(f"accepted stem does not exist: {stem}")


def load_smoke_listening_bundle(
    audit_manifest: Path,
    receipt_path: Path,
    *,
    project_root: Path,
    require_artifacts: bool = True,
) -> SmokeListeningBundle:
    """Load and cross-check the item audit against the aggregate receipt."""

    root = project_root.resolve()
    rows = _load_jsonl(audit_manifest)
    receipt = _load_json(receipt_path)
    expected = receipt.get("input_items")
    if not isinstance(expected, int) or expected < 0:
        raise SmokeListeningContractError("receipt input_items must be a non-negative integer")
    if len(rows) != expected:
        raise SmokeListeningContractError(
            f"receipt expects {expected} items but audit has {len(rows)}"
        )

    item_ids: set[str] = set()
    for row in rows:
        _validate_row(row, project_root=root, require_artifacts=require_artifacts)
        item_id = str(row["item_id"])
        if item_id in item_ids:
            raise SmokeListeningContractError(f"duplicate item_id: {item_id}")
        item_ids.add(item_id)

    actual_counts = {
        tier: sum(row["acceptance_tier"] == tier for row in rows)
        for tier in ("gold", "silver", "rejected")
    }
    receipt_counts = receipt.get("acceptance_counts")
    if not isinstance(receipt_counts, dict) or any(
        receipt_counts.get(tier) != count for tier, count in actual_counts.items()
    ):
        raise SmokeListeningContractError(
            f"receipt acceptance_counts {receipt_counts!r} disagree with {actual_counts!r}"
        )
    if receipt.get("accepted_items") != actual_counts["gold"] + actual_counts["silver"]:
        raise SmokeListeningContractError("receipt accepted_items disagrees with audit")
    if receipt.get("rejected_items") != actual_counts["rejected"]:
        raise SmokeListeningContractError("receipt rejected_items disagrees with audit")

    return SmokeListeningBundle(tuple(rows), receipt, root)


def filter_smoke_records(
    records: Sequence[Mapping[str, Any]],
    *,
    tiers: Iterable[str] | None = None,
    routes: Iterable[str] | None = None,
    metadata_splits: Iterable[str] | None = None,
    label_query: str = "",
) -> list[Mapping[str, Any]]:
    """Filter audit rows; an empty filter collection means no restriction."""

    tier_set = set(tiers or ())
    route_set = set(routes or ())
    split_set = set(metadata_splits or ())
    query = label_query.strip().casefold()
    selected = [
        row
        for row in records
        if (not tier_set or str(row["acceptance_tier"]) in tier_set)
        and (not route_set or str(row["materialization_source_route"]) in route_set)
        and (not split_set or str(row["metadata_split"]) in split_set)
        and (
            not query
            or query in str(row["label"]).casefold()
            or query in str(row["canonical_display_name"]).casefold()
        )
    ]
    return sorted(
        selected,
        key=lambda row: (
            TIER_ORDER[str(row["acceptance_tier"])],
            str(row["label"]).casefold(),
            str(row["item_id"]),
        ),
    )


CHECK_EXPLANATIONS = {
    "stem_rms_dbfs": "RMS của stem quá nhỏ; âm tách có thể gần như không nghe thấy.",
    "stem_peak_dbfs": "Đỉnh biên độ stem quá nhỏ; sự kiện thiếu mức âm rõ ràng.",
    "retained_energy_ratio_min": "Stem giữ quá ít năng lượng so với source crop.",
    "retained_energy_ratio_max": "Stem có năng lượng bất thường cao so với source crop.",
    "target_text_similarity": "Stem chưa khớp đủ mạnh với nhãn âm thanh cần tách.",
    "target_residual_margin": "Nhãn target còn hiện diện trong residual quá nhiều so với stem.",
    "target_other_margin": "Stem khớp một nhãn gây nhiễu mạnh hơn hoặc quá gần nhãn target.",
    "paraphrase_agreement": "Hai cách viết prompt tạo kết quả chưa đủ nhất quán.",
    "clipped_sample_fraction": "Tỷ lệ sample bị clipping vượt giới hạn.",
    "upstream_ambiguity": "Crop có target khác chồng lấn (tier 2/3), nên không được nhận ở mức Silver.",
}


def _condition_detail(
    code: str,
    metrics: Mapping[str, Any],
    level: Mapping[str, Any],
    *,
    gate_name: str,
) -> dict[str, Any]:
    mapping = {
        "stem_rms_dbfs": ("stem_rms_dbfs", "minimum_rms_dbfs", ">="),
        "stem_peak_dbfs": ("stem_peak_dbfs", "minimum_peak_dbfs", ">="),
        "retained_energy_ratio_min": (
            "retained_energy_ratio",
            "minimum_retained_energy_ratio",
            ">=",
        ),
        "retained_energy_ratio_max": (
            "retained_energy_ratio",
            "maximum_retained_energy_ratio",
            "<=",
        ),
        "target_text_similarity": (
            "target_text_similarity",
            "minimum_target_text_similarity",
            ">=",
        ),
        "target_residual_margin": (
            "target_residual_margin",
            "minimum_target_residual_margin",
            ">=",
        ),
        "target_other_margin": (
            "target_other_margin",
            "minimum_target_other_margin",
            ">=",
        ),
        "paraphrase_agreement": (
            "paraphrase_agreement",
            "minimum_paraphrase_agreement",
            ">=",
        ),
        "clipped_sample_fraction": (
            "maximum_clipped_sample_fraction",
            "maximum_clipped_sample_fraction",
            "<=",
        ),
    }
    if code not in mapping:
        return {
            "code": code,
            "gate": gate_name,
            "value": None,
            "operator": "",
            "threshold": None,
            "explanation": CHECK_EXPLANATIONS.get(code, "Quality check không đạt."),
        }
    metric_key, threshold_key, operator = mapping[code]
    return {
        "code": code,
        "gate": gate_name,
        "value": metrics.get(metric_key),
        "operator": operator,
        "threshold": level.get(threshold_key),
        "explanation": CHECK_EXPLANATIONS[code],
    }


def rejection_details(
    record: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return the exact minimum-acceptance conditions that rejected a row."""

    if record.get("acceptance_tier") != "rejected":
        return []
    gate_result = record["quality_gate"]
    metrics = record["quality_metrics"]
    levels = receipt["quality_gate"]
    details = [
        _condition_detail(code, metrics, levels["silver"], gate_name="silver")
        for code in gate_result.get("silver_failures", [])
    ]
    if gate_result.get("silver_disallowed_by_upstream_ambiguity"):
        details.append(
            {
                "code": "upstream_ambiguity",
                "gate": "policy",
                "value": record.get("ambiguity_tier"),
                "operator": "<=",
                "threshold": 1,
                "explanation": CHECK_EXPLANATIONS["upstream_ambiguity"],
            }
        )
        details.extend(
            _condition_detail(code, metrics, levels["gold"], gate_name="gold")
            for code in gate_result.get("gold_failures", [])
        )
    return details


def not_gold_details(
    record: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Explain why an accepted Silver item did not meet Gold thresholds."""

    if record.get("acceptance_tier") != "silver":
        return []
    return [
        _condition_detail(
            code,
            record["quality_metrics"],
            receipt["quality_gate"]["gold"],
            gate_name="gold",
        )
        for code in record["quality_gate"].get("gold_failures", [])
    ]


def metric_table_rows(
    record: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build compact, presentation-ready rows for the eight gate metrics."""

    metrics = record["quality_metrics"]
    gold = receipt["quality_gate"]["gold"]
    silver = receipt["quality_gate"]["silver"]
    definitions = (
        (
            "Stem RMS ↑",
            "stem_rms_dbfs",
            f">= {gold['minimum_rms_dbfs']:.3g}",
            f">= {silver['minimum_rms_dbfs']:.3g}",
            "Độ lớn trung bình, dBFS",
        ),
        (
            "Stem peak ↑",
            "stem_peak_dbfs",
            f">= {gold['minimum_peak_dbfs']:.3g}",
            f">= {silver['minimum_peak_dbfs']:.3g}",
            "Đỉnh biên độ, dBFS",
        ),
        (
            "Retained energy ↔",
            "retained_energy_ratio",
            f"{gold['minimum_retained_energy_ratio']:.3g}–{gold['maximum_retained_energy_ratio']:.3g}",
            f"{silver['minimum_retained_energy_ratio']:.3g}–{silver['maximum_retained_energy_ratio']:.3g}",
            "Năng lượng stem / source",
        ),
        (
            "Target-text similarity ↑",
            "target_text_similarity",
            f">= {gold['minimum_target_text_similarity']:.3g}",
            f">= {silver['minimum_target_text_similarity']:.3g}",
            "Mức khớp stem với nhãn target",
        ),
        (
            "Target-residual margin ↑",
            "target_residual_margin",
            f">= {gold['minimum_target_residual_margin']:.3g}",
            f">= {silver['minimum_target_residual_margin']:.3g}",
            "Target ở stem trội hơn residual",
        ),
        (
            "Target-other margin ↑",
            "target_other_margin",
            f">= {gold['minimum_target_other_margin']:.3g}",
            f">= {silver['minimum_target_other_margin']:.3g}",
            "Target trội hơn nhãn gây nhiễu",
        ),
        (
            "Paraphrase agreement ↑",
            "paraphrase_agreement",
            f">= {gold['minimum_paraphrase_agreement']:.3g}",
            f">= {silver['minimum_paraphrase_agreement']:.3g}",
            "Hai prompt phải cho stem nhất quán",
        ),
        (
            "Clipped fraction ↓",
            "maximum_clipped_sample_fraction",
            f"<= {gold['maximum_clipped_sample_fraction']:.3g}",
            f"<= {silver['maximum_clipped_sample_fraction']:.3g}",
            "Tỷ lệ sample clipping lớn nhất",
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, key, gold_text, silver_text, explanation in definitions:
        value = metrics.get(key)
        rows.append(
            {
                "Metric": name,
                "Giá trị": "N/A" if value is None else f"{float(value):.4f}",
                "Gold": gold_text,
                "Silver": silver_text,
                "Ý nghĩa": explanation,
            }
        )
    return rows


def format_interval(onset: Any, offset: Any) -> str:
    return f"{float(onset):.3f}–{float(offset):.3f} s"

