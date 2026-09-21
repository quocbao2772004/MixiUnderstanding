from pathlib import Path

import torch

from mixi_understanding.scripts.evaluate_qces_oracle_component_ceiling_v1 import (
    ComponentRow,
    build_variants,
    condition_summary,
    decision_from_summary,
    rank_result,
)


def _row() -> ComponentRow:
    return ComponentRow(
        event_id="scene:e00",
        scene_id="scene",
        label="a",
        label_id=0,
        component_path=Path("unused.wav"),
        onset_frame=2,
        offset_frame=4,
        onset_sample=1280,
        offset_sample=2560,
        cleanliness_tier="gold",
        duration_equalized=False,
        recipe_kind="pair_overlap",
        requested_overlap_fraction=0.5,
        maximum_concurrency=2,
        source_id="source",
        source_sha256="hash",
    )


def test_build_variants_preserves_component_and_alignment() -> None:
    component = torch.linspace(-1.0, 1.0, 1280)
    isolated, aligned = build_variants(component, _row())
    assert isolated.shape == aligned.shape == (160000,)
    assert torch.equal(isolated[:1280], component)
    assert torch.count_nonzero(isolated[1280:]) == 0
    assert torch.equal(aligned[1280:2560], component)
    assert torch.count_nonzero(aligned[:1280]) == 0
    assert torch.count_nonzero(aligned[2560:]) == 0


def test_rank_and_condition_summary_use_gold_only_for_scoring() -> None:
    labels = ["a", "b", "c"]
    first = rank_result(torch.tensor([1.0, 3.0, 2.0]), 0, labels)
    second = rank_result(torch.tensor([3.0, 2.0, 1.0]), 0, labels)
    assert first["rank"] == 3
    assert first["predicted_label"] == "b"
    assert second["rank"] == 1
    rows = [{"isolated": first}, {"isolated": second}]
    summary = condition_summary(rows, "isolated")
    assert summary["top1_accuracy_↑"] == 0.5
    assert summary["top5_accuracy_↑"] == 1.0


def test_decision_selects_separator_only_with_recognizable_sources() -> None:
    def metrics(top1: float, top5: float) -> dict[str, float]:
        return {"top1_accuracy_↑": top1, "top5_accuracy_↑": top5}

    separator = decision_from_summary(
        {
            "isolated": metrics(0.90, 0.97),
            "aligned": metrics(0.88, 0.96),
            "mixture": metrics(0.40, 0.70),
        }
    )
    assert separator["recommendation"] == "separator_first"
    data = decision_from_summary(
        {
            "isolated": metrics(0.70, 0.85),
            "aligned": metrics(0.69, 0.84),
            "mixture": metrics(0.30, 0.60),
        }
    )
    assert data["recommendation"] == "repair_data_or_ontology"
