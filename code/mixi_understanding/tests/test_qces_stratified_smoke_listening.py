from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import pytest

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "qces/stratified_smoke_listening.py"
MODULE_NAME = "qces_stratified_smoke_listening_contract_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, CONTRACT_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = CONTRACT
SPEC.loader.exec_module(CONTRACT)

SmokeListeningContractError = CONTRACT.SmokeListeningContractError
filter_smoke_records = CONTRACT.filter_smoke_records
load_smoke_listening_bundle = CONTRACT.load_smoke_listening_bundle
metric_table_rows = CONTRACT.metric_table_rows
rejection_details = CONTRACT.rejection_details


def _receipt(counts: dict[str, int]) -> dict:
    return {
        "input_items": sum(counts.values()),
        "accepted_items": counts["gold"] + counts["silver"],
        "rejected_items": counts["rejected"],
        "acceptance_counts": counts,
        "quality_gate": {
            "gold": {
                "minimum_rms_dbfs": -45.0,
                "minimum_peak_dbfs": -35.0,
                "minimum_retained_energy_ratio": 0.005,
                "maximum_retained_energy_ratio": 2.5,
                "minimum_target_text_similarity": 0.15,
                "minimum_target_residual_margin": 0.02,
                "minimum_target_other_margin": 0.0,
                "minimum_paraphrase_agreement": 0.75,
                "maximum_clipped_sample_fraction": 0.001,
            },
            "silver": {
                "minimum_rms_dbfs": -55.0,
                "minimum_peak_dbfs": -45.0,
                "minimum_retained_energy_ratio": 0.001,
                "maximum_retained_energy_ratio": 4.0,
                "minimum_target_text_similarity": 0.05,
                "minimum_target_residual_margin": -0.05,
                "minimum_target_other_margin": -0.08,
                "minimum_paraphrase_agreement": 0.55,
                "maximum_clipped_sample_fraction": 0.005,
            },
        },
    }


def _row(tmp_path: Path, *, label: str, tier: str) -> dict:
    source = tmp_path / f"{label}.source.flac"
    source.write_bytes(b"source")
    stem = tmp_path / f"{label}.stem.flac"
    accepted = tier != "rejected"
    if accepted:
        stem.write_bytes(b"stem")
    return {
        "acceptance_tier": tier,
        "accepted": accepted,
        "active_offset_seconds": 0.8,
        "active_onset_seconds": 0.2,
        "ambiguity_tier": 0,
        "canonical_display_name": label,
        "canonical_prompt": label,
        "duration_seconds": 1.0,
        "item_id": f"item-{label}",
        "label": label,
        "materialization_source_route": "route-a",
        "metadata_split": "train",
        "quality_gate": {
            "gold_failures": [] if tier == "gold" else ["target_text_similarity"],
            "silver_failures": ["target_text_similarity"] if tier == "rejected" else [],
            "silver_disallowed_by_upstream_ambiguity": False,
        },
        "quality_metrics": {
            "stem_rms_dbfs": -30.0,
            "stem_peak_dbfs": -20.0,
            "retained_energy_ratio": 0.5,
            "target_text_similarity": 0.01 if tier == "rejected" else 0.10,
            "target_residual_margin": 0.1,
            "target_other_margin": None,
            "paraphrase_agreement": 0.9,
            "maximum_clipped_sample_fraction": 0.0,
        },
        "source_audio_path": str(source),
        "source_crop_end_seconds": 3.0,
        "source_crop_start_seconds": 2.0,
        "source_split": "train",
        "source_video_id": "video",
        "stem_path": str(stem) if accepted else "",
        "strong_annotation_offset_seconds": 0.8,
        "strong_annotation_onset_seconds": 0.2,
    }


def _write_bundle(tmp_path: Path, rows: list[dict], receipt: dict) -> tuple[Path, Path]:
    audit = tmp_path / "audit.jsonl"
    audit.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return audit, receipt_path


def test_bundle_contract_and_default_rejected_filter(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, label="Rejected", tier="rejected"),
        _row(tmp_path, label="Silver", tier="silver"),
        _row(tmp_path, label="Gold", tier="gold"),
    ]
    audit, receipt_path = _write_bundle(
        tmp_path, rows, _receipt({"gold": 1, "silver": 1, "rejected": 1})
    )
    bundle = load_smoke_listening_bundle(
        audit, receipt_path, project_root=tmp_path, require_artifacts=True
    )
    assert bundle.acceptance_counts == {"gold": 1, "silver": 1, "rejected": 1}
    selected = filter_smoke_records(bundle.records, tiers=["rejected"])
    assert [row["label"] for row in selected] == ["Rejected"]
    assert len(filter_smoke_records(bundle.records, tiers=[])) == 3


def test_rejection_detail_uses_silver_threshold_and_metric_table_has_arrows(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path, label="Tick", tier="rejected")
    receipt = _receipt({"gold": 0, "silver": 0, "rejected": 1})
    details = rejection_details(row, receipt)
    assert details == [
        {
            "code": "target_text_similarity",
            "gate": "silver",
            "value": 0.01,
            "operator": ">=",
            "threshold": 0.05,
            "explanation": "Stem chưa khớp đủ mạnh với nhãn âm thanh cần tách.",
        }
    ]
    metric_names = [item["Metric"] for item in metric_table_rows(row, receipt)]
    assert len(metric_names) == 8
    assert any("↑" in name for name in metric_names)
    assert any("↓" in name for name in metric_names)
    assert any("↔" in name for name in metric_names)


def test_ambiguity_policy_is_reported_as_rejection_reason(tmp_path: Path) -> None:
    row = _row(tmp_path, label="Bathtub", tier="rejected")
    row["ambiguity_tier"] = 3
    row["quality_gate"] = {
        "gold_failures": ["target_other_margin"],
        "silver_failures": [],
        "silver_disallowed_by_upstream_ambiguity": True,
    }
    row["quality_metrics"]["target_other_margin"] = -0.04
    details = rejection_details(
        row, _receipt({"gold": 0, "silver": 0, "rejected": 1})
    )
    assert [detail["code"] for detail in details] == [
        "upstream_ambiguity",
        "target_other_margin",
    ]
    assert details[1]["gate"] == "gold"


def test_receipt_count_mismatch_fails_closed(tmp_path: Path) -> None:
    row = _row(tmp_path, label="Gold", tier="gold")
    audit, receipt_path = _write_bundle(
        tmp_path, [row], _receipt({"gold": 0, "silver": 0, "rejected": 1})
    )
    with pytest.raises(SmokeListeningContractError, match="acceptance_counts"):
        load_smoke_listening_bundle(
            audit, receipt_path, project_root=tmp_path, require_artifacts=True
        )
