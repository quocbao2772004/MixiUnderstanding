from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mixi_understanding.data.qces_v5_tacos import (
    CC0_SPDX,
    CC0_URL,
    DOI,
    OFFICIAL_FILES,
    PACKET_FORMAT,
    PLAN_FORMAT,
    COMPLIANCE_FORMAT,
    QUESTION_SLOTS,
    RECORD_ID,
    SOURCE_DURATION_SAFETY_SAMPLES,
    SOURCE_DURATION_SAFETY_SECONDS,
    Candidate,
    ChainProposal,
    Region,
    TacosAuditError,
    assign_creator_hash_rank_partitions,
    build_cross_scene_absent_anchor_pools,
    canonical_license,
    rank_candidate_chains,
    select_balanced_candidates,
    valid_window_start_sample_interval,
    validate_record_json,
)


def candidate(
    index: int,
    subclass: str,
    *,
    creator: str | None = None,
    overlap: bool = False,
    defective: bool = False,
) -> Candidate:
    regions = tuple(
        Region(
            region_id=f"region_{i:03d}",
            caption=f"Distinct event {index} number {i}",
            onset=float(i),
            offset=float(i) + (1.5 if overlap else 0.5),
            eligible=True,
            exclusion_reasons=(),
        )
        for i in range(5)
    )
    proposal = ChainProposal(
        region_ids=tuple(region.region_id for region in regions[:4]),
        pairwise_cosines=(0.0, 0.1, 0.2, 0.0, 0.1, 0.0),
        max_pairwise_cosine=0.2,
        mean_pairwise_cosine=0.066667,
        minimum_adjacent_onset_gap=1.0,
        first_onset=0.0,
        last_offset=3.5,
        valid_window_start_sample_low=0,
        valid_window_start_sample_high=0,
        window_start_sample=0,
        window_end_sample=320_000,
        chain_rank_tie_sha256="1" * 64,
        window_start_sha256="2" * 64,
        ranked_chain_count=5,
        semantically_diverse_chain_count=5,
    )
    return Candidate(
        filename=f"{1000 + index}.mp3",
        freesound_id=str(1000 + index),
        sound_link=f"https://freesound.org/people/u{index}/sounds/{1000 + index}/",
        creator_name=creator or f"creator-{index}",
        creator_id=f"creator-key-{creator or index}",
        superclass="Sound",
        subclass=subclass,
        upstream_split="development" if index % 2 == 0 else "test",
        audio_license_spdx=CC0_SPDX,
        audio_license_url=CC0_URL,
        original_duration=30.0,
        crop_start=0.0,
        crop_end=10.01,
        regions=regions,
        source_metadata_reasons=("defect",) if defective else (),
        best_chain=proposal,
        ranked_chain_count=5,
        semantically_diverse_chain_count=5,
    )


class TacosLicenseAndRecordTest(unittest.TestCase):
    def test_primary_policy_accepts_cc0_but_not_nc_or_sampling(self) -> None:
        self.assertEqual(
            canonical_license("http://creativecommons.org/publicdomain/zero/1.0/"),
            (CC0_SPDX, CC0_URL),
        )
        self.assertIsNone(
            canonical_license("https://creativecommons.org/licenses/by-nc/4.0/")
        )
        self.assertIsNone(
            canonical_license("https://creativecommons.org/licenses/sampling+/1.0/")
        )

    def test_record_descriptor_is_independently_pinned(self) -> None:
        payload = {
            "id": RECORD_ID,
            "created": "2025-05-12T14:45:28Z",
            "updated": "2025-05-13T10:48:33Z",
            "metadata": {"doi": DOI},
            "files": [
                {"key": key, "size": size, "checksum": f"md5:{checksum}"}
                for key, (size, checksum) in OFFICIAL_FILES.items()
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            receipt = validate_record_json(path)
            self.assertEqual(receipt["record_id"], RECORD_ID)
            payload["files"][0]["size"] += 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TacosAuditError, "inventory"):
                validate_record_json(path)


class TacosSelectionAndQuestionContractTest(unittest.TestCase):
    def test_absent_pool_reuses_exact_positive_captions_from_other_scenes(self) -> None:
        selected = [candidate(index, f"class-{index}") for index in range(16)]
        scene_partitions = {
            f"tacos_{item.freesound_id}": ("real_dev" if index < 8 else "real_test")
            for index, item in enumerate(selected)
        }
        scene_tiers = {
            f"tacos_{item.freesound_id}": ("reserve" if index in {7, 15} else "core")
            for index, item in enumerate(selected)
        }
        pools, audit = build_cross_scene_absent_anchor_pools(
            selected,
            scene_partitions=scene_partitions,
            scene_tiers=scene_tiers,
            seed=2026,
        )
        reversed_pools, reversed_audit = build_cross_scene_absent_anchor_pools(
            list(reversed(selected)),
            scene_partitions=scene_partitions,
            scene_tiers=scene_tiers,
            seed=2026,
        )
        self.assertEqual(pools, reversed_pools)
        self.assertEqual(audit, reversed_audit)
        positive_by_scene = {
            f"tacos_{item.freesound_id}": {
                region.caption
                for region in item.regions
                if region.region_id in item.suggested_region_ids
            }
            for item in selected
        }
        for scene_id, pool in pools.items():
            self.assertEqual(len(pool["absent_anchor_candidates"]), 6)
            self.assertEqual(
                pool["absent_anchor_candidates"],
                [row["caption"] for row in pool["cross_scene_support"]],
            )
            for support in pool["cross_scene_support"]:
                self.assertEqual(
                    support["selection_partition"], scene_partitions[scene_id]
                )
                self.assertEqual(support["support_tier"], "core")
                self.assertNotIn(support["caption"], positive_by_scene[scene_id])
                self.assertNotIn(scene_id, support["cross_scene_support_scene_ids"])
                self.assertTrue(support["cross_scene_support_scene_ids"])
                self.assertTrue(
                    all(
                        support["caption"] in positive_by_scene[support_scene]
                        and scene_partitions[support_scene]
                        == scene_partitions[scene_id]
                        and scene_tiers[support_scene] == "core"
                        for support_scene in support["cross_scene_support_scene_ids"]
                    )
                )
            self.assertEqual(
                len(
                    {
                        support["primary_support_scene_id"]
                        for support in pool["cross_scene_support"]
                    }
                ),
                6,
            )
        self.assertEqual(audit["source"], "other_selected_chain_captions")
        self.assertEqual(audit["unsupported_negative_captions_↓"], 0)
        self.assertEqual(audit["cross_partition_support_violations_↓"], 0)
        self.assertEqual(audit["negative_positive_vocabulary_support_fraction_↑"], 1.0)
        self.assertFalse(audit["qces_label_selection_used"])

    def test_absent_pool_fails_closed_without_six_cross_scene_captions(self) -> None:
        with self.assertRaisesRegex(TacosAuditError, "too few distinct core"):
            build_cross_scene_absent_anchor_pools(
                [candidate(0, "A"), candidate(1, "B")],
                scene_partitions={
                    "tacos_1000": "real_dev",
                    "tacos_1001": "real_dev",
                },
                scene_tiers={"tacos_1000": "core", "tacos_1001": "core"},
                seed=2026,
            )

    def test_material_anti_shortcut_contract_uses_v3_schemas(self) -> None:
        self.assertTrue(PLAN_FORMAT.endswith("_v3"))
        self.assertTrue(COMPLIANCE_FORMAT.endswith("_v3"))
        self.assertTrue(PACKET_FORMAT.endswith("_v3"))

    def test_quantized_valid_interval_uses_true_max_offset(self) -> None:
        regions = (
            Region("r0", "a", 4.0, 5.0, True, ()),
            Region("r1", "b", 5.0, 6.0, True, ()),
            Region("r2", "c", 6.0, 14.2, True, ()),
            Region("r3", "d", 7.0, 8.0, True, ()),
        )
        interval = valid_window_start_sample_interval(
            regions, clip_duration=23.2, window_seconds=10.0
        )
        # max offset=14.2 forces low=4.2 s, while first onset=4.0 caps high;
        # therefore no 10 s crop can contain this chain.
        self.assertIsNone(interval)

        contained = (
            regions[0],
            regions[1],
            Region("r2", "c", 6.0, 13.5, True, ()),
            regions[3],
        )
        self.assertEqual(
            valid_window_start_sample_interval(
                contained, clip_duration=23.2, window_seconds=10.0
            ),
            (112_000, 128_000),
        )

    def test_valid_interval_reserves_frozen_source_duration_margin(self) -> None:
        regions = (
            Region("r0", "a", 10.0, 10.5, True, ()),
            Region("r1", "b", 11.0, 11.5, True, ()),
            Region("r2", "c", 12.0, 12.5, True, ()),
            Region("r3", "d", 13.0, 13.5, True, ()),
        )
        duration = 627_744 / 32_000
        interval = valid_window_start_sample_interval(
            regions, clip_duration=duration, window_seconds=10.0
        )
        assert interval is not None
        _low, high = interval
        self.assertEqual(SOURCE_DURATION_SAFETY_SECONDS, 0.01)
        self.assertEqual(SOURCE_DURATION_SAFETY_SAMPLES, 320)
        self.assertEqual(high, 307_424)
        self.assertLessEqual(high + 320_000, 627_744 - 320)
        self.assertNotEqual(high, 307_744)

    def test_window_truncated_region_is_explicitly_ineligible(self) -> None:
        region = Region(
            region_id="region_000",
            caption="Boundary event",
            onset=4.5,
            offset=5.5,
            eligible=True,
            exclusion_reasons=(),
        )
        shifted = region.as_shifted_dict(
            window_start=5.0,
            window_end=15.0,
            chain_region_ids=frozenset(),
        )
        assert shifted is not None
        self.assertTrue(shifted["truncated_by_benchmark_window"])
        self.assertFalse(shifted["eligible_proposal"])
        self.assertIn(
            "truncated_by_benchmark_window",
            shifted["proposal_exclusion_reasons"],
        )

    def test_semantic_chain_is_exactly_four_and_inside_hash_uniform_window(
        self,
    ) -> None:
        base = candidate(20, "A")
        captions = [
            "distinct event 20 number 0",
            "distinct event 20 number 1",
            "distinct event 20 number 2",
            "distinct event 20 number 3",
            "distinct event 20 number 4",
        ]
        embeddings = {
            caption: tuple(float(index == axis) for axis in range(5))
            for index, caption in enumerate(captions)
        }
        ranked = rank_candidate_chains(
            base,
            embeddings=embeddings,
            seed=2026,
            minimum_onset_gap_seconds=0.35,
            benchmark_window_seconds=10.0,
            maximum_pairwise_cosine=0.80,
            cosine_round_decimals=6,
        )
        self.assertIsNotNone(ranked.best_chain)
        assert ranked.best_chain is not None
        self.assertEqual(len(ranked.best_chain.region_ids), 4)
        self.assertEqual(ranked.best_chain.window_start_sample, 0)
        self.assertEqual(ranked.best_chain.window_end_sample, 320_000)
        self.assertLessEqual(ranked.best_chain.max_pairwise_cosine, 0.80)
        self.assertEqual(len(ranked.best_chain.chain_rank_tie_sha256), 64)
        self.assertEqual(len(ranked.best_chain.window_start_sha256), 64)

    def test_semantic_threshold_fails_closed_without_relaxation(self) -> None:
        base = candidate(21, "A")
        base = Candidate(**{**base.__dict__, "regions": base.regions[:4]})
        captions = [
            normalize
            for normalize in (
                "distinct event 21 number 0",
                "distinct event 21 number 1",
                "distinct event 21 number 2",
                "distinct event 21 number 3",
            )
        ]
        embeddings = {caption: (1.0, 0.0) for caption in captions}
        ranked = rank_candidate_chains(
            base,
            embeddings=embeddings,
            seed=2026,
            minimum_onset_gap_seconds=0.35,
            benchmark_window_seconds=10.0,
            maximum_pairwise_cosine=0.80,
            cosine_round_decimals=6,
        )
        self.assertIsNone(ranked.best_chain)
        self.assertGreater(ranked.ranked_chain_count, 0)
        self.assertEqual(ranked.semantically_diverse_chain_count, 0)

    def test_creator_hash_rank_split_is_exact_stable_and_disjoint(self) -> None:
        creators = [f"creator-key-{index}" for index in range(156)]
        first, first_fingerprint = assign_creator_hash_rank_partitions(creators)
        second, second_fingerprint = assign_creator_hash_rank_partitions(
            reversed(creators)
        )
        self.assertEqual(first, second)
        self.assertEqual(first_fingerprint, second_fingerprint)
        self.assertEqual(
            sum(row["partition"] == "real_dev" for row in first.values()), 31
        )
        self.assertEqual(
            sum(row["partition"] == "real_test" for row in first.values()), 125
        )
        self.assertEqual(
            sorted(row["hash_rank"] for row in first.values()), list(range(156))
        )
        self.assertTrue(all(len(str(row["sha256"])) == 64 for row in first.values()))

    def test_selection_is_deterministic_balanced_and_creator_disjoint(self) -> None:
        pool = [
            candidate(0, "A", overlap=True),
            candidate(1, "A"),
            candidate(2, "B", overlap=True),
            candidate(3, "B"),
            candidate(4, "C", overlap=True),
            candidate(5, "C"),
            candidate(6, "D", defective=True),
        ]
        first = select_balanced_candidates(
            pool,
            count=3,
            seed=2026,
            split_name="real_test",
            overlap_fraction=0.5,
        )
        second = select_balanced_candidates(
            list(reversed(pool)),
            count=3,
            seed=2026,
            split_name="real_test",
            overlap_fraction=0.5,
        )
        self.assertEqual(
            [item.filename for item in first], [item.filename for item in second]
        )
        self.assertEqual(len({item.subclass for item in first}), 3)
        self.assertEqual(len({item.creator_id for item in first}), 3)
        self.assertNotIn("D", {item.subclass for item in first})

        forbidden = {first[0].creator_id}
        other = select_balanced_candidates(
            pool,
            count=2,
            seed=2026,
            split_name="real_dev",
            forbidden_creator_ids=forbidden,
        )
        self.assertTrue(forbidden.isdisjoint({item.creator_id for item in other}))

    def test_reused_creator_cannot_satisfy_unique_scene_target(self) -> None:
        pool = [candidate(0, "A", creator="same"), candidate(1, "B", creator="same")]
        with self.assertRaisesRegex(TacosAuditError, "unique creators"):
            select_balanced_candidates(pool, count=2, seed=2026, split_name="real_test")

    def test_fixed_contract_has_eight_questions_and_balanced_controls(self) -> None:
        self.assertEqual(len(QUESTION_SLOTS), 8)
        self.assertEqual(sum(bool(slot["no_evidence"]) for slot in QUESTION_SLOTS), 2)
        self.assertEqual(
            {slot["relation"] for slot in QUESTION_SLOTS},
            {"after", "before", "first"},
        )
        self.assertTrue(
            all(
                slot["anchor_selector"] == "human_verified_absent_anchor"
                for slot in QUESTION_SLOTS
                if slot["no_evidence"]
            )
        )
        first_slots = [slot for slot in QUESTION_SLOTS if slot["relation"] == "first"]
        self.assertEqual(
            {slot["option_order_variant"] for slot in first_slots},
            {"forward", "reversed"},
        )


if __name__ == "__main__":
    unittest.main()
