from __future__ import annotations

import json
from pathlib import Path

from mixi_understanding.qces.acoustic_cleanliness import (
    MaterializedAudio,
    audit_video_events,
    best_event_per_label_video,
    build_crop_source_plan,
    interval_union_length,
    load_materialized_audio,
    summarize_class_support,
)
from mixi_understanding.qces.supported_ontology import StrongEvent


def _event(
    mid: str,
    label: str,
    onset: float,
    offset: float,
    *,
    video: str = "video",
) -> StrongEvent:
    return StrongEvent(
        segment_id=f"{video}_00000",
        video_id=video,
        mid=mid,
        label=label,
        display_name=label,
        onset_seconds=onset,
        offset_seconds=offset,
    )


def test_interval_union_does_not_double_count() -> None:
    assert interval_union_length([(1.0, 2.0), (1.5, 3.0), (4.0, 4.5)]) == 2.5


def test_audit_separates_all_selected_and_hierarchy_overlap() -> None:
    rows = audit_video_events(
        metadata_split="train",
        video_id="video",
        events=[
            _event("/a", "A", 1.0, 3.0),
            _event("/b", "B", 2.0, 2.5),
            _event("/parent", "Parent", 2.4, 4.0),
        ],
        selected_mids={"/a", "/b"},
        hierarchy_descendants={"/parent": {"/a"}},
    )
    target = next(row for row in rows if row["mid"] == "/a")
    assert target["any_different_overlap_seconds"] == 1.0
    assert target["any_different_overlap_fraction"] == 0.5
    assert target["selected_distractor_overlap_seconds"] == 0.5
    assert abs(target["hierarchy_related_overlap_seconds"] - 0.6) < 1e-9
    assert target["overlapping_selected_distractor_mids"] == ["/b"]
    assert target["ambiguity_tier"] == 3
    assert not target["fully_isolated"]


def test_explicit_ambiguity_tiers_and_margins() -> None:
    isolated = audit_video_events(
        metadata_split="train",
        video_id="isolated",
        events=[
            _event("/a", "A", 1.0, 2.0, video="isolated"),
            _event("/x", "X", 0.0, 0.5, video="isolated"),
            _event("/x", "X", 2.5, 3.0, video="isolated"),
        ],
        selected_mids={"/a"},
    )[0]
    assert isolated["left_isolation_margin_seconds"] == 0.5
    assert isolated["right_isolation_margin_seconds"] == 0.5
    assert isolated["ambiguity_tier_name"] == "fully_isolated_all_strong_labels"

    tight = audit_video_events(
        metadata_split="train",
        video_id="tight",
        events=[
            _event("/a", "A", 1.0, 2.0, video="tight"),
            _event("/x", "X", 2.1, 3.0, video="tight"),
        ],
        selected_mids={"/a"},
    )[0]
    assert tight["ambiguity_tier_name"] == "selected_ontology_isolated"

    low = audit_video_events(
        metadata_split="train",
        video_id="low",
        events=[
            _event("/a", "A", 1.0, 2.0, video="low"),
            _event("/x", "X", 1.95, 3.0, video="low"),
        ],
        selected_mids={"/a"},
        maximum_low_overlap_fraction=0.10,
    )[0]
    assert low["ambiguity_tier_name"] == "selected_ontology_isolated"
    assert low["clean_source_eligible"]

    low_selected = audit_video_events(
        metadata_split="train",
        video_id="low_selected",
        events=[
            _event("/a", "A", 1.0, 2.0, video="low_selected"),
            _event("/b", "B", 1.95, 3.0, video="low_selected"),
        ],
        selected_mids={"/a", "/b"},
        maximum_low_overlap_fraction=0.10,
    )[0]
    assert low_selected["ambiguity_tier_name"] == "low_selected_distractor_overlap"


def test_plan_prioritizes_cleanliness_before_materialization_and_emits_context() -> None:
    video_events = {
        "download": [
            _event("/a", "A", 1.0, 2.0, video="download"),
        ],
        "local": [
            _event("/a", "A", 1.0, 2.0, video="local"),
            _event("/b", "B", 1.95, 2.2, video="local"),
        ],
    }
    rows = []
    rows.extend(
        audit_video_events(
            metadata_split="train",
            video_id="download",
            events=video_events["download"],
            selected_mids={"/a", "/b"},
        )
    )
    rows.extend(
        audit_video_events(
            metadata_split="train",
            video_id="local",
            events=video_events["local"],
            selected_mids={"/a", "/b"},
            materialized=MaterializedAudio(
                video_id="local",
                metadata_split="train",
                audio_path="local.flac",
                audio_exists=True,
                protocol_splits=("dev",),
            ),
        )
    )
    best = best_event_per_label_video(rows, seed=7)
    plan, summary = build_crop_source_plan(
        best_candidates=best,
        events_by_split={"train": video_events, "eval": {}},
        selected_mids={"/a", "/b"},
        target_train_videos_per_class=1,
        target_eval_videos_per_class=1,
        seed=7,
    )
    a_row = next(row for row in plan if row["coverage_label"] == "A")
    assert a_row["video_id"] == "download"  # strict tier 0 beats local tier 2
    assert "target_label" not in a_row
    assert a_row["strong_annotations"][0]["mid"] == "/a"
    assert not summary["leakage_controls"]["uses_qa_questions_or_answers"]


def test_class_support_counts_unique_videos_not_events() -> None:
    base = {
        "metadata_split": "train",
        "label": "A",
        "mid": "/a",
        "ambiguity_tier": 0,
        "selected_ontology_isolated": True,
        "any_different_overlap_fraction": 0.0,
        "materialized_audio_exists": False,
        "duration_seconds": 1.0,
    }
    best = {
        ("train", "A", "v1"): {**base, "video_id": "v1"},
        ("train", "A", "v2"): {**base, "video_id": "v2"},
        ("eval", "A", "e1"): {
            **base,
            "metadata_split": "eval",
            "video_id": "e1",
            "ambiguity_tier": 2,
            "selected_ontology_isolated": False,
        },
    }
    rows = summarize_class_support(
        selected_rows=[{"mid": "/a", "label": "A", "display_name": "A"}],
        best_candidates=best,
        train_target=2,
        eval_target=1,
    )
    assert rows[0]["raw_train_unique_videos"] == 2
    assert rows[0]["isolated_train_unique_videos"] == 2
    assert rows[0]["isolated_eval_unique_videos"] == 0
    assert not rows[0]["isolated_100_20_feasible"]
    assert rows[0]["clean_100_20_feasible"]


def test_materialized_manifest_preserves_upstream_and_protocol_splits(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "video_id": "v1",
            "protocol_upstream_split": "train",
            "protocol_split": "dev",
            "mixture_path": str(audio),
            "audio_sha256": "abc",
        },
        {
            "video_id": "v1",
            "hf_split": "train",
            "split": "dev",
            "mixture_path": str(audio),
            "audio_sha256": "abc",
        },
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = load_materialized_audio([manifest], project_root=tmp_path)
    item = loaded[("train", "v1")]
    assert item.audio_exists
    assert item.protocol_splits == ("dev",)
