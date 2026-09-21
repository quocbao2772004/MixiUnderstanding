#!/usr/bin/env python3
"""Audit a QCES AudioSep dual-role oracle report without rerunning AudioSep.

The report contains one row per question, but several questions can request the
same pair of event stems from the same scene.  This audit therefore also
reports one value per unique acoustic comparison
``(scene_id, unordered evidence_event_ids)`` and labels its bootstrap interval
as conditional on the observed smoke family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from mixi_understanding.qces.data import QCESManifestDataset


UNION_GATED = "union_text__oracle_union_window"
DUAL_GATED = "dual_role_text__oracle_role_windows_linear_add"
UNION_RAW = "union_text__ungated"
DUAL_RAW = "dual_role_text__ungated_linear_add"

AUDITED_METRICS = {
    "evidence_sd_sdr_db_↑": 1.0,
    "evidence_si_sdr_db_↑": 1.0,
    "evidence_l1_↓": -1.0,
    "oracle_windowed_weakest_role_sd_sdr_db_↑": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip target-stem/window invariants (metadata audit still runs).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def acoustic_key(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(row["scene_id"]), tuple(sorted(map(str, row["evidence_event_ids"])))


def role_windows_overlap(row: Mapping[str, Any]) -> bool:
    return any(
        max(float(a[0]), float(b[0])) < min(float(a[1]), float(b[1]))
        for a in row["anchor_intervals"]
        for b in row["answer_intervals"]
    )


def deduplicate_acoustic_units(
    items: Sequence[Mapping[str, Any]], rows_by_id: Mapping[str, Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    grouped: dict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[acoustic_key(rows_by_id[str(item["id"])])].append(item)
    representatives: list[Mapping[str, Any]] = []
    for key, group in grouped.items():
        representative = group[0]
        # Anchor/answer metrics may exchange places when the same acoustic pair
        # appears in before and after questions.  Evidence and weakest-role
        # metrics must nevertheless be identical for one acoustic comparison.
        for other in group[1:]:
            for mode in (UNION_GATED, DUAL_GATED, UNION_RAW, DUAL_RAW):
                for metric in AUDITED_METRICS:
                    left = float(representative["metrics"][mode][metric])
                    right = float(other["metrics"][mode][metric])
                    if abs(left - right) > 1e-7:
                        raise ValueError(
                            f"duplicate acoustic unit {key} disagrees on {mode}/{metric}"
                        )
        representatives.append(representative)
    return representatives


def summarize_pair(
    items: Sequence[Mapping[str, Any]],
    union_mode: str,
    dual_mode: str,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot summarize an empty subset")
    result: dict[str, Any] = {"count": len(items), "metrics": {}}
    for metric_index, (metric, sign) in enumerate(AUDITED_METRICS.items()):
        union = np.asarray(
            [float(item["metrics"][union_mode][metric]) for item in items]
        )
        dual = np.asarray(
            [float(item["metrics"][dual_mode][metric]) for item in items]
        )
        advantage = sign * (dual - union)
        rng = np.random.default_rng(seed + metric_index)
        indices = rng.integers(
            0, len(advantage), size=(bootstrap_replicates, len(advantage))
        )
        bootstrap_means = advantage[indices].mean(axis=1)
        result["metrics"][metric] = {
            "union_mean": float(union.mean()),
            "dual_mean": float(dual.mean()),
            "dual_advantage_mean_↑": float(advantage.mean()),
            "dual_advantage_median_↑": float(np.median(advantage)),
            "dual_advantage_p10_↑": float(np.quantile(advantage, 0.10)),
            "dual_advantage_minimum_↑": float(advantage.min()),
            "dual_win_rate_↑": float((advantage > 0.0).mean()),
            "conditional_bootstrap_95ci_mean_↑": [
                float(value)
                for value in np.quantile(bootstrap_means, (0.025, 0.975))
            ],
        }
    return result


def _subset(
    items: Iterable[Mapping[str, Any]], field: str, value: Any
) -> list[Mapping[str, Any]]:
    return [item for item in items if item[field] == value]


def audit_audio_targets(
    manifest: Path, item_ids: set[str]
) -> dict[str, float | int]:
    dataset = QCESManifestDataset(manifest, crop_samples=None)
    maxima = {
        "evidence_minus_anchor_plus_answer_max_abs_↓": 0.0,
        "residual_minus_mixture_minus_evidence_max_abs_↓": 0.0,
        "anchor_energy_outside_anchor_window_max_abs_↓": 0.0,
        "answer_energy_outside_answer_window_max_abs_↓": 0.0,
        "evidence_energy_outside_union_window_max_abs_↓": 0.0,
    }
    audited = 0
    for index, record in enumerate(dataset.records):
        if record.sample_id not in item_ids:
            continue
        example = dataset[index]
        union = torch.maximum(example.anchor_mask, example.answer_mask)
        maxima["evidence_minus_anchor_plus_answer_max_abs_↓"] = max(
            maxima["evidence_minus_anchor_plus_answer_max_abs_↓"],
            float((example.evidence - example.anchor_stem - example.answer_stem).abs().max()),
        )
        maxima["residual_minus_mixture_minus_evidence_max_abs_↓"] = max(
            maxima["residual_minus_mixture_minus_evidence_max_abs_↓"],
            float((example.residual - (example.mixture - example.evidence)).abs().max()),
        )
        maxima["anchor_energy_outside_anchor_window_max_abs_↓"] = max(
            maxima["anchor_energy_outside_anchor_window_max_abs_↓"],
            float((example.anchor_stem * (1.0 - example.anchor_mask)).abs().max()),
        )
        maxima["answer_energy_outside_answer_window_max_abs_↓"] = max(
            maxima["answer_energy_outside_answer_window_max_abs_↓"],
            float((example.answer_stem * (1.0 - example.answer_mask)).abs().max()),
        )
        maxima["evidence_energy_outside_union_window_max_abs_↓"] = max(
            maxima["evidence_energy_outside_union_window_max_abs_↓"],
            float((example.evidence * (1.0 - union)).abs().max()),
        )
        audited += 1
    return {"audited_records": audited, **maxima}


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise SystemExit("--bootstrap-replicates must be positive")
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text())
    manifest = Path(report["manifest"]).resolve()
    rows = read_jsonl(manifest)
    rows_by_id = {str(row["id"]): row for row in rows}
    items = list(report["items"])
    item_ids = {str(item["id"]) for item in items}
    answerable_ids = {str(row["id"]) for row in rows if not row["no_evidence"]}
    if item_ids != answerable_ids:
        raise ValueError("report items do not equal the manifest answerable set")

    if sha256_file(manifest) != report["manifest_sha256"]:
        raise ValueError("manifest SHA-256 mismatch")
    for path_key, hash_key in (
        ("audiosep_config", "audiosep_config_sha256"),
        ("audiosep_checkpoint", "audiosep_checkpoint_sha256"),
    ):
        if sha256_file(Path(report[path_key])) != report[hash_key]:
            raise ValueError(f"{path_key} SHA-256 mismatch")

    for item in items:
        row = rows_by_id[str(item["id"])]
        events = {str(event["event_id"]): event for event in row["events"]}
        anchor = events[str(row["anchor_event_ids"][0])]
        answer = events[str(row["answer_event_ids"][0])]
        checks = {
            "scene_id": str(row["scene_id"]),
            "relation": str(row["relation"]),
            "anchor_label": str(anchor["label"]),
            "answer_label": str(answer["label"]),
            "same_role_label": str(anchor["label"]) == str(answer["label"]),
            "role_windows_overlap": role_windows_overlap(row),
        }
        for field, expected in checks.items():
            if item[field] != expected:
                raise ValueError(f"item {item['id']} disagrees with manifest field {field}")
        if item["anchor_intervals_seconds"] != row["anchor_intervals"]:
            raise ValueError(f"item {item['id']} anchor intervals disagree")
        if item["answer_intervals_seconds"] != row["answer_intervals"]:
            raise ValueError(f"item {item['id']} answer intervals disagree")
        if bool(item["same_role_prompt"]) != bool(item["same_role_label"]):
            raise ValueError(f"item {item['id']} label/prompt equality disagrees")

    expected_cached_calls = 0
    active_mixture: str | None = None
    active_prompts: set[str] = set()
    for item in items:
        mixture_path = str(rows_by_id[str(item["id"])]["mixture_path"])
        if mixture_path != active_mixture:
            active_mixture = mixture_path
            active_prompts = set()
        for prompt in (item["union_prompt"], item["anchor_prompt"], item["answer_prompt"]):
            if prompt not in active_prompts:
                expected_cached_calls += 1
                active_prompts.add(str(prompt))
    reported_cached_calls = int(
        report["counts"]["raw_separator_calls_with_consecutive_scene_prompt_cache"]
    )
    if expected_cached_calls != reported_cached_calls:
        raise ValueError("reported cache call count is inconsistent")

    units = deduplicate_acoustic_units(items, rows_by_id)

    def summaries(source: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "primary": summarize_pair(
                source,
                UNION_GATED,
                DUAL_GATED,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            ),
            "secondary_ungated": summarize_pair(
                source,
                UNION_RAW,
                DUAL_RAW,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            ),
        }

    receipt: dict[str, Any] = {
        "format": "qces_dual_role_oracle_independent_audit_v1",
        "scope": "internal_v5_smoke_only_not_paper_evidence",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "integrity": {
            "manifest_and_checkpoint_hashes_match": True,
            "answerable_item_set_matches": True,
            "metadata_fields_and_windows_match": True,
            "expected_cached_separator_calls": expected_cached_calls,
            "reported_cached_separator_calls": reported_cached_calls,
        },
        "dependence": {
            "question_rows": len(items),
            "unique_acoustic_comparison_units": len(units),
            "unique_counterfactual_scenes": len({item["scene_id"] for item in items}),
            "unique_scene_families": len(
                {rows_by_id[str(item["id"])]["scene_family_id"] for item in items}
            ),
            "acoustic_key": "(scene_id, unordered evidence_event_ids)",
            "uncertainty_warning": (
                "Bootstrap intervals are conditional sensitivity intervals over "
                "the observed acoustic units. One scene family is insufficient "
                "for a scene/family-generalization confidence interval."
            ),
        },
        "question_row_analysis": summaries(items),
        "acoustic_unit_analysis": {
            "overall": summaries(units),
            "by_relation": {},
            "by_role_windows_overlap": {},
            "by_same_role_label": {},
            "distinct_label_by_role_windows_overlap": {},
        },
    }
    for relation in sorted({str(item["relation"]) for item in items}):
        subset = deduplicate_acoustic_units(
            _subset(items, "relation", relation), rows_by_id
        )
        receipt["acoustic_unit_analysis"]["by_relation"][relation] = summaries(subset)
    for value in (False, True):
        subset = deduplicate_acoustic_units(
            _subset(items, "role_windows_overlap", value), rows_by_id
        )
        receipt["acoustic_unit_analysis"]["by_role_windows_overlap"][str(value)] = summaries(subset)
        subset = deduplicate_acoustic_units(
            _subset(items, "same_role_label", value), rows_by_id
        )
        receipt["acoustic_unit_analysis"]["by_same_role_label"][str(value)] = summaries(subset)
        distinct_overlap = [
            item
            for item in items
            if not item["same_role_label"] and item["role_windows_overlap"] == value
        ]
        distinct_overlap = deduplicate_acoustic_units(distinct_overlap, rows_by_id)
        receipt["acoustic_unit_analysis"][
            "distinct_label_by_role_windows_overlap"
        ][str(value)] = summaries(distinct_overlap)
    if not args.skip_audio:
        receipt["audio_target_invariants"] = audit_audio_targets(manifest, item_ids)

    output = args.output or report_path.with_name("independent_audit_receipt.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["dependence"], indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
