"""Checkpoint-free tests for the dual-role AudioSep oracle diagnostic."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.scripts.evaluate_audiosep_dual_role_oracle import (
    DUAL_GATED,
    DUAL_RAW,
    DUAL_UNION_GATED,
    UNION_GATED,
    UNION_RAW,
    compose_modes,
    evaluate_mode,
    oracle_prompts,
    paint_intervals,
    render_unique_prompts,
    separator_call_provenance,
    summarize_items,
    summarize_unique_acoustic_items,
)


class _ConditionScaledSeparator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, batch):  # type: ignore[no-untyped-def]
        self.calls += 1
        scale = batch["condition"][:, :1, None]
        return {"waveform": batch["mixture"] * scale}


class PromptAndRenderingTest(unittest.TestCase):
    def test_union_prompt_matches_timeline_order_not_role_id_order(self) -> None:
        record = Mock(spec=QCESV5Record)
        record.no_evidence = False
        record.anchor_event_ids = ("late_anchor",)
        record.answer_event_ids = ("early_answer",)
        record.evidence_event_ids = ("late_anchor", "early_answer")
        events = {
            "late_anchor": SimpleNamespace(
                event_id="late_anchor", label="Buzz", onset_seconds=2.0
            ),
            "early_answer": SimpleNamespace(
                event_id="early_answer", label="Croak", onset_seconds=0.5
            ),
        }
        record.event_by_id.side_effect = events.__getitem__

        prompts = oracle_prompts(record)

        self.assertEqual(prompts.union, "a frog croaking and a buzzing sound")
        self.assertEqual(prompts.anchor, "a buzzing sound")
        self.assertEqual(prompts.answer, "a frog croaking")
        self.assertFalse(prompts.same_role_prompt)

    def test_unique_prompt_rendering_reuses_identical_role_prompt(self) -> None:
        separator = _ConditionScaledSeparator()
        mixture = torch.ones(4)
        conditions = {
            "bell and bell": torch.tensor([0.5]),
            "bell": torch.tensor([0.25]),
        }

        rendered, calls = render_unique_prompts(
            separator,
            mixture,
            conditions,
            ("bell and bell", "bell", "bell"),
        )

        self.assertEqual(calls, 2)
        self.assertEqual(separator.calls, 2)
        self.assertEqual(set(rendered), {"bell and bell", "bell"})
        self.assertTrue(torch.equal(rendered["bell"], torch.full((4,), 0.25)))

    def test_distinct_roles_add_only_inside_their_aligned_windows(self) -> None:
        union = torch.full((4,), 5.0)
        anchor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        answer = torch.tensor([10.0, 20.0, 30.0, 40.0])
        anchor_gate = torch.tensor([1.0, 1.0, 1.0, 0.0])
        answer_gate = torch.tensor([0.0, 1.0, 1.0, 1.0])

        modes = compose_modes(
            union,
            anchor,
            answer,
            anchor_gate,
            answer_gate,
            same_role_prompt=False,
        )

        self.assertTrue(
            torch.equal(modes[DUAL_RAW].evidence, anchor + answer)
        )
        self.assertTrue(
            torch.equal(
                modes[DUAL_UNION_GATED].evidence,
                torch.tensor([11.0, 22.0, 33.0, 44.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                modes[DUAL_GATED].evidence,
                torch.tensor([1.0, 22.0, 33.0, 40.0]),
            )
        )
        self.assertTrue(
            torch.equal(modes[UNION_GATED].evidence, union)
        )
        self.assertTrue(
            torch.equal(
                modes[DUAL_GATED].oracle_windowed_anchor,
                torch.tensor([1.0, 2.0, 3.0, 0.0]),
            )
        )

    def test_same_prompt_is_never_doubled_in_overlap(self) -> None:
        raw = torch.tensor([1.0, 2.0, 3.0, 4.0])
        anchor_gate = torch.tensor([1.0, 1.0, 1.0, 0.0])
        answer_gate = torch.tensor([0.0, 1.0, 1.0, 1.0])

        modes = compose_modes(
            raw,
            raw,
            raw,
            anchor_gate,
            answer_gate,
            same_role_prompt=True,
        )

        self.assertTrue(torch.equal(modes[DUAL_RAW].evidence, raw))
        self.assertTrue(torch.equal(modes[DUAL_UNION_GATED].evidence, raw))
        self.assertTrue(torch.equal(modes[DUAL_GATED].evidence, raw))
        self.assertEqual(float(modes[DUAL_GATED].evidence[1]), 2.0)

    def test_factorial_b_holds_union_window_fixed_while_c_aligns_roles(self) -> None:
        union = torch.full((6,), 5.0)
        anchor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        answer = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        anchor_gate = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        answer_gate = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])

        modes = compose_modes(
            union,
            anchor,
            answer,
            anchor_gate,
            answer_gate,
            same_role_prompt=False,
        )

        self.assertTrue(
            torch.equal(
                modes[UNION_GATED].evidence,
                torch.tensor([5.0, 5.0, 0.0, 5.0, 5.0, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                modes[DUAL_UNION_GATED].evidence,
                torch.tensor([11.0, 22.0, 0.0, 44.0, 55.0, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                modes[DUAL_GATED].evidence,
                torch.tensor([1.0, 2.0, 0.0, 40.0, 50.0, 0.0]),
            )
        )

    def test_interval_painting_uses_union_without_role_overwrite(self) -> None:
        anchor = paint_intervals(((0.0, 0.3),), 10, 5)
        answer = paint_intervals(((0.2, 0.5),), 10, 5)

        self.assertTrue(torch.equal(anchor, torch.tensor([1, 1, 1, 0, 0])))
        self.assertTrue(torch.equal(answer, torch.tensor([0, 0, 1, 1, 1])))
        self.assertEqual(float(torch.maximum(anchor, answer).sum()), 5.0)


class MetricProtocolTest(unittest.TestCase):
    def test_residual_is_arithmetic_complement_and_summary_has_arrows(self) -> None:
        mixture = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 0.25])
        anchor = torch.tensor([0.0, 0.6, -0.4, 0.0, 0.0, 0.0])
        answer = torch.tensor([0.0, 0.0, -0.3, 0.4, -0.2, 0.0])
        target = anchor + answer
        target_residual = mixture - target
        gate_anchor = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        gate_answer = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        modes = compose_modes(
            target,
            anchor,
            answer,
            gate_anchor,
            gate_answer,
            same_role_prompt=False,
        )
        metrics = {
            mode: evaluate_mode(
                rendered,
                mixture,
                target,
                target_residual,
                anchor,
                answer,
            )
            for mode, rendered in modes.items()
        }
        item = {
            "relation": "after",
            "role_windows_overlap": True,
            "same_role_label": False,
            "metrics": metrics,
        }

        summary = summarize_items([item])

        for mode in (
            UNION_RAW,
            DUAL_RAW,
            UNION_GATED,
            DUAL_UNION_GATED,
            DUAL_GATED,
        ):
            self.assertEqual(metrics[mode]["mixture_consistency_l1_sanity_↓"], 0.0)
            self.assertAlmostEqual(
                metrics[mode]["evidence_l1_↓"],
                metrics[mode]["residual_l1_↓"],
                places=7,
            )
        primary = summary["overall"][
            "primary_oracle_windowed_dual_advantage"
        ]
        self.assertTrue(primary)
        self.assertTrue(all(key.endswith("↑") for key in primary))
        for mode_summary in summary["overall"]["modes"].values():
            self.assertTrue(
                all(key.endswith(("↑", "↓")) for key in mode_summary)
            )
        factorial = summary["overall"]["factorial_matched_temporal"]
        self.assertEqual(factorial["A_mode"], UNION_GATED)
        self.assertEqual(factorial["B_mode"], DUAL_UNION_GATED)
        self.assertEqual(factorial["C_mode"], DUAL_GATED)
        for comparison in (
            "A_to_B_role_factorized_semantic_extraction_advantage",
            "B_to_C_role_window_alignment_advantage",
            "A_to_C_total_advantage",
        ):
            self.assertTrue(factorial[comparison])
            self.assertTrue(
                all(key.endswith("↑") for key in factorial[comparison])
            )

    def test_unique_acoustic_summary_deduplicates_surface_and_relation_rows(self) -> None:
        base_metrics = {
            mode: {
                f"{metric}_{direction}": 1.0
                for metric, direction in {
                    "evidence_si_sdr_db": "↑",
                    "evidence_sd_sdr_db": "↑",
                    "evidence_si_sdri_db": "↑",
                    "evidence_sd_sdri_db": "↑",
                    "evidence_l1": "↓",
                    "residual_l1": "↓",
                    "oracle_windowed_anchor_si_sdr_db": "↑",
                    "oracle_windowed_answer_si_sdr_db": "↑",
                    "oracle_windowed_weakest_role_si_sdr_db": "↑",
                    "oracle_windowed_anchor_sd_sdr_db": "↑",
                    "oracle_windowed_answer_sd_sdr_db": "↑",
                    "oracle_windowed_weakest_role_sd_sdr_db": "↑",
                    "mixture_consistency_l1_sanity": "↓",
                }.items()
            }
            for mode in (
                UNION_RAW,
                DUAL_RAW,
                UNION_GATED,
                DUAL_UNION_GATED,
                DUAL_GATED,
            )
        }
        rows = [
            {
                "id": "q0",
                "scene_id": "scene0",
                "evidence_event_ids": ["e0", "e1"],
                "relation": "after",
                "role_windows_overlap": True,
                "same_role_label": False,
                "metrics": base_metrics,
            },
            {
                "id": "q1",
                "scene_id": "scene0",
                "evidence_event_ids": ["e1", "e0"],
                "relation": "before",
                "role_windows_overlap": True,
                "same_role_label": False,
                "metrics": base_metrics,
            },
        ]

        summary = summarize_unique_acoustic_items(rows)

        self.assertEqual(summary["overall"]["count"], 1)
        self.assertEqual(summary["overall"]["question_row_count"], 2)
        self.assertEqual(summary["by_relation"]["after"]["count"], 1)
        self.assertEqual(summary["by_relation"]["before"]["count"], 1)
        self.assertNotIn(
            "oracle_windowed_anchor_sd_sdr_db_mean_↑",
            summary["overall"]["modes"][UNION_GATED],
        )

    def test_call_provenance_separates_effective_and_physical_calls(self) -> None:
        items = [
            {
                "union_prompt": "bell and bell",
                "anchor_prompt": "bell",
                "answer_prompt": "bell",
            },
            {
                "union_prompt": "buzz and croak",
                "anchor_prompt": "buzz",
                "answer_prompt": "croak",
            },
        ]

        provenance = separator_call_provenance(items, physical_cached_calls=4)

        effective = provenance[
            "effective_calls_if_mode_deployed_independently_no_inter_record_cache"
        ]
        physical = provenance["physical_calls_for_joint_factorial_evaluator"]
        self.assertEqual(effective["A_union_semantic_union_window"], 2)
        self.assertEqual(effective["B_dual_semantic_union_window"], 3)
        self.assertEqual(effective["C_dual_semantic_role_windows"], 3)
        self.assertEqual(
            physical["without_inter_record_cache_B_and_C_share_raw_stems"], 5
        )
        self.assertEqual(
            physical[
                "with_consecutive_scene_prompt_cache_B_and_C_share_raw_stems"
            ],
            4,
        )


if __name__ == "__main__":
    unittest.main()
