#!/usr/bin/env python3
"""Design-dev-only audit for revising the QCES 188-class hierarchy.

No model is trained here.  The frozen target-invariant checkpoint is evaluated
on design-dev mixture and correctly aligned clean components.  Merge candidates
require acoustic confusion plus proximity in the official AudioSet ontology.
Matched predictions are never read; the matched manifest is used only to block
identities while auditing availability of a future untouched evaluation set.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import torch

from mixi_understanding.qces.clean_evidence_scenes import _atomic_write_text
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    load_scene_manifest,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _sha256_file
from mixi_understanding.scripts.train_qces_v5_oracle_group_experts_v1 import (
    group_lookup,
    ordering_metrics,
)
from mixi_understanding.scripts.train_qces_v5_target_invariant_atst_v1 import (
    TargetInvariantExperts,
    blocked_eval_identities,
    build_backbone,
    component_rows,
    evaluate_rows,
    read_jsonl,
    resolved_audio_path,
    stripped_eval,
)


FORMAT = "qces_v5_hierarchy_readiness_v1"
ALIASES = {
    "Blender_and_food_processor": "Blender",
    "Glass_chink_and_clink": "Chink, clink",
    "Glass_shatter": "Shatter",
    "Gurgling_and_bubbling": "Gurgling",
    "Stream_and_river": "Stream",
    "Tire_squeal_and_skidding": "Tire squeal",
}


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    run = base / "v5_target_invariant_atst_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=run / "target_invariant_atst_v1_best.pt")
    parser.add_argument("--previous-experts", type=Path, default=base / "v5_oracle_group_experts_v1/oracle_group_experts_v1_best.pt")
    parser.add_argument("--dev-manifest", type=Path, default=Path("/var/tmp/qces_full188_tiered_realistic_v5/detector_scene_manifest_tiered_dev.jsonl"))
    parser.add_argument("--matched-manifest", type=Path, default=Path("/var/tmp/qces_v5_matched_eval_v1/detector_scene_manifest_matched_eval.jsonl"))
    parser.add_argument("--accepted-index", type=Path, default=PROJECT_ROOT / "data_full/index/accepted_samples.jsonl")
    parser.add_argument("--ontology-json", type=Path, default=PROJECT_ROOT / "outputs/qces_supported_ontology_official_assets_v1/audioset_ontology.json")
    parser.add_argument("--class-index", type=Path, default=PRETRAINED_SED_ROOT / "hf_dataset_gen/metadata/class_labels_indices.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "v5_hierarchy_readiness_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_name(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.replace("_", " ").replace("&", " and ").lower())
    return " ".join(token for token in tokens if token != "and")


def label_mids(labels: Sequence[str], class_index: Path) -> dict[int, str]:
    rows = list(csv.DictReader(class_index.resolve().open("r", encoding="utf-8")))
    by_name = {normalized_name(str(row["display_name"])): str(row["mid"]) for row in rows}
    result: dict[int, str] = {}
    missing: list[str] = []
    for label_id, label in enumerate(labels):
        display = ALIASES.get(label, label)
        mid = by_name.get(normalized_name(display))
        if mid is None:
            missing.append(label)
        else:
            result[label_id] = mid
    if missing:
        raise RuntimeError(f"official ontology mapping incomplete: {missing}")
    return result


class OfficialHierarchy:
    def __init__(self, path: Path) -> None:
        rows = json.loads(path.resolve().read_text(encoding="utf-8"))
        self.name = {str(row["id"]): str(row["name"]) for row in rows}
        self.parents: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            parent = str(row["id"])
            for child in row.get("child_ids", []):
                self.parents[str(child)].add(parent)

    def distances(self, node: str) -> dict[str, int]:
        result = {node: 0}
        queue = deque([node])
        while queue:
            value = queue.popleft()
            for parent in self.parents.get(value, set()):
                if parent not in result or result[parent] > result[value] + 1:
                    result[parent] = result[value] + 1
                    queue.append(parent)
        return result

    def relation(self, left: str, right: str) -> dict[str, Any]:
        left_distance = self.distances(left)
        right_distance = self.distances(right)
        common = set(left_distance) & set(right_distance)
        common.discard(left); common.discard(right)
        if not common:
            return {"lca_mid": None, "lca_name": None, "ontology_distance": None, "same_immediate_parent": False}
        lca = min(common, key=lambda value: (left_distance[value] + right_distance[value], max(left_distance[value], right_distance[value])))
        immediate = bool(self.parents.get(left, set()) & self.parents.get(right, set()))
        return {
            "lca_mid": lca,
            "lca_name": self.name.get(lca, lca),
            "ontology_distance": int(left_distance[lca] + right_distance[lca]),
            "same_immediate_parent": immediate,
        }


def confusion(ordering: torch.Tensor, targets: torch.Tensor, classes: int) -> torch.Tensor:
    matrix = torch.zeros((classes, classes), dtype=torch.long)
    for gold, predicted in zip(targets.tolist(), ordering[:, 0].tolist(), strict=True):
        matrix[int(gold), int(predicted)] += 1
    return matrix


def safe_accuracy(matrix: torch.Tensor, label: int) -> float:
    total = int(matrix[label].sum())
    return int(matrix[label, label]) / max(total, 1)


def alias_components(num_classes: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(num_classes))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges:
        union(left, right)
    grouped: dict[int, list[int]] = defaultdict(list)
    for value in range(num_classes):
        grouped[find(value)].append(value)
    return sorted(grouped.values(), key=lambda values: (values[0], len(values)))


def collapsed_accuracy(ordering: torch.Tensor, targets: torch.Tensor, components: Sequence[Sequence[int]]) -> float:
    lookup = torch.empty(188, dtype=torch.long)
    for cluster_id, component in enumerate(components):
        lookup[torch.tensor(component)] = cluster_id
    return float(lookup[ordering[:, 0]].eq(lookup[targets]).float().mean())


def future_eval_availability(
    accepted_index: Path,
    matched_manifest: Path,
    labels: Sequence[str],
) -> dict[str, Any]:
    blocked = blocked_eval_identities([matched_manifest.resolve()])
    label_set = set(labels)
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for row in read_jsonl(accepted_index):
        label = str(row.get("label") or "")
        if label not in label_set or str(row.get("split") or "") != "eval" or str(row.get("quality_tier") or "") != "gold":
            continue
        source_id = str(row.get("sample_id") or "")
        video_id = str(row.get("video_id") or "")
        path = resolved_audio_path(str(row.get("audio_path") or "")).as_posix()
        if not source_id or source_id in seen:
            continue
        if source_id in blocked["source_id"] or (video_id and video_id in blocked["video_id"]) or path in blocked["source_path"]:
            continue
        counts[label] += 1
        seen.add(source_id)
    missing = [label for label in labels if counts[label] == 0]
    checks = {
        "covers_all_188_classes": len(counts) == 188,
        "at_least_5_sources_per_class": len(counts) == 188 and min(counts.values()) >= 5,
    }
    return {
        "unused_gold_eval_sources": sum(counts.values()),
        "covered_classes": len(counts),
        "missing_classes": missing,
        "minimum_nonzero_per_class": min(counts.values()) if counts else 0,
        "median_nonzero_per_class": float(np.median(list(counts.values()))) if counts else 0.0,
        "per_class": dict(sorted(counts.items())),
        "gates": {"passed": all(checks.values()), "checks": checks},
        "decision": "sufficient_to_lock_new_test" if all(checks.values()) else "acquire_new_disjoint_eval_sources_before_next_model_claim",
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path = path.resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = list(payload["labels"])
    groups = [list(map(int, group)) for group in payload["groups"]]
    label_to_group, _ = group_lookup(groups, len(labels))
    label_to_id = {label: index for index, label in enumerate(labels)}
    previous = torch.load(args.previous_experts.resolve(), map_location="cpu", weights_only=False)
    device = make_device(args.device)
    backbone = build_backbone(len(labels), device)
    state = backbone.model.state_dict(); state.update(payload["atst_trainable_state_dict"])
    backbone.model.load_state_dict(state, strict=True)
    model = TargetInvariantExperts(
        int(payload["input_dim"]), int(payload["hidden_dim"]), groups, previous["model_state_dict"]
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    dev_rows = load_scene_manifest(args.dev_manifest.resolve(), label_to_id)
    mixture = evaluate_rows(
        backbone, model, dev_rows, groups, label_to_group, device,
        batch_size=args.batch_size, num_workers=args.num_workers, amp=args.amp,
    )
    clean = evaluate_rows(
        backbone, model, component_rows(dev_rows), groups, label_to_group, device,
        batch_size=args.batch_size, num_workers=args.num_workers, amp=args.amp,
        component_canvas=True,
    )
    if not torch.equal(mixture["targets"], clean["targets"]):
        raise RuntimeError("dev mixture/clean event alignment differs")
    targets = mixture["targets"]
    matrices = {
        "clean_oracle": confusion(clean["oracle_order"], targets, len(labels)),
        "mixture_oracle": confusion(mixture["oracle_order"], targets, len(labels)),
        "mixture_routed": confusion(mixture["routed_order"], targets, len(labels)),
    }
    mids = label_mids(labels, args.class_index)
    hierarchy = OfficialHierarchy(args.ontology_json)
    pair_rows: list[dict[str, Any]] = []
    high_confidence_edges: list[tuple[int, int]] = []
    support = matrices["clean_oracle"].sum(1)
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            clean_count = int(matrices["clean_oracle"][left, right] + matrices["clean_oracle"][right, left])
            mixture_count = int(matrices["mixture_oracle"][left, right] + matrices["mixture_oracle"][right, left])
            clean_rate = clean_count / max(int(support[left] + support[right]), 1)
            left_direction = int(matrices["clean_oracle"][left, right]) / max(int(support[left]), 1)
            right_direction = int(matrices["clean_oracle"][right, left]) / max(int(support[right]), 1)
            if clean_count < 2 or (clean_rate < 0.08 and max(left_direction, right_direction) < 0.25):
                continue
            relation = hierarchy.relation(mids[left], mids[right])
            high_confidence = bool(
                relation["same_immediate_parent"]
                and clean_rate >= 0.15
                and int(matrices["clean_oracle"][left, right]) > 0
                and int(matrices["clean_oracle"][right, left]) > 0
            )
            if high_confidence:
                high_confidence_edges.append((left, right))
            pair_rows.append({
                "left_label_id": left,
                "left_label": labels[left],
                "right_label_id": right,
                "right_label": labels[right],
                "clean_left_to_right": int(matrices["clean_oracle"][left, right]),
                "clean_right_to_left": int(matrices["clean_oracle"][right, left]),
                "clean_mutual_confusion_rate_↑": clean_rate,
                "mixture_mutual_confusion_count": mixture_count,
                "same_immediate_parent": relation["same_immediate_parent"],
                "lca_name": relation["lca_name"],
                "ontology_distance_↓": relation["ontology_distance"],
                "recommendation": "merge_alias_listening_review" if high_confidence else "data_or_boundary_review",
            })
    pair_rows.sort(
        key=lambda row: (
            row["recommendation"] != "merge_alias_listening_review",
            -float(row["clean_mutual_confusion_rate_↑"]),
            -int(row["mixture_mutual_confusion_count"]),
        )
    )
    components = alias_components(len(labels), high_confidence_edges)
    nontrivial = [component for component in components if len(component) > 1]
    member_to_cluster = {
        member: cluster_id for cluster_id, component in enumerate(nontrivial) for member in component
    }
    class_rows: list[dict[str, Any]] = []
    true_group = label_to_group[targets]
    for label_id, label in enumerate(labels):
        selected = targets == label_id
        clean_accuracy = safe_accuracy(matrices["clean_oracle"], label_id)
        mix_oracle_accuracy = safe_accuracy(matrices["mixture_oracle"], label_id)
        routed_accuracy = safe_accuracy(matrices["mixture_routed"], label_id)
        router_accuracy = float(mixture["predicted_groups"][selected].eq(true_group[selected]).float().mean())
        clean_row = matrices["clean_oracle"][label_id].clone(); clean_row[label_id] = -1
        mix_row = matrices["mixture_oracle"][label_id].clone(); mix_row[label_id] = -1
        clean_confused = int(clean_row.argmax()); mix_confused = int(mix_row.argmax())
        if label_id in member_to_cluster:
            action = "merge_alias_listening_review"
        elif clean_accuracy < 0.50:
            action = "audit_sources_and_label_definition"
        elif clean_accuracy - mix_oracle_accuracy >= 0.15:
            action = "keep_label_improve_overlap_robustness"
        elif router_accuracy < 0.70:
            action = "keep_label_improve_coarse_router"
        else:
            action = "keep"
        class_rows.append({
            "label_id": label_id,
            "label": label,
            "support_events": int(selected.sum()),
            "clean_oracle_top1_↑": clean_accuracy,
            "mixture_oracle_top1_↑": mix_oracle_accuracy,
            "mixture_routed_top1_↑": routed_accuracy,
            "clean_minus_mixture_gap_↓": clean_accuracy - mix_oracle_accuracy,
            "router_group_accuracy_↑": router_accuracy,
            "top_clean_confusion": labels[clean_confused],
            "top_clean_confusion_count": max(0, int(clean_row[clean_confused])),
            "top_mixture_confusion": labels[mix_confused],
            "top_mixture_confusion_count": max(0, int(mix_row[mix_confused])),
            "proposed_action": action,
        })
    class_rows.sort(key=lambda row: (float(row["clean_oracle_top1_↑"]), float(row["mixture_oracle_top1_↑"]), row["label"]))
    collapsed = {
        "output_concepts_after_high_confidence_aliases": len(components),
        "nontrivial_alias_clusters": len(nontrivial),
        "labels_in_alias_clusters": sum(map(len, nontrivial)),
        "clean_oracle_collapsed_top1_↑": collapsed_accuracy(clean["oracle_order"], targets, components),
        "mixture_oracle_collapsed_top1_↑": collapsed_accuracy(mixture["oracle_order"], targets, components),
        "mixture_routed_collapsed_top1_↑": collapsed_accuracy(mixture["routed_order"], targets, components),
    }
    flattened_events: list[dict[str, Any]] = []
    for scene in dev_rows:
        for event in scene.events:
            if str(event.get("event_kind", "semantic")) != "semantic":
                continue
            flattened_events.append({
                **dict(event),
                "scene_id": scene.scene_id,
                "mixture_path": scene.mixture_path,
            })
    if len(flattened_events) != len(targets):
        raise RuntimeError("flattened dev event order/count differs from model evaluation")
    clean_prediction = clean["oracle_order"][:, 0]
    mixture_prediction = mixture["oracle_order"][:, 0]
    listening_rows: list[dict[str, Any]] = []
    for pair_id, (left, right) in enumerate(high_confidence_edges):
        pair_indices = [index for index, target in enumerate(targets.tolist()) if target in {left, right}]
        buckets: dict[str, list[int]] = defaultdict(list)
        for index in pair_indices:
            gold = int(targets[index]); predicted = int(clean_prediction[index])
            if predicted == (right if gold == left else left):
                kind = "bidirectional_confusion"
            elif predicted == gold:
                kind = "correct_control"
            else:
                continue
            direction = f"{gold}_to_{predicted}"
            if len(buckets[f"{kind}:{direction}"]) < 3:
                buckets[f"{kind}:{direction}"].append(index)
        selected_indices = [index for key in sorted(buckets) for index in buckets[key]]
        for example_index, index in enumerate(selected_indices):
            event = flattened_events[index]
            gold = int(targets[index]); predicted = int(clean_prediction[index])
            kind = "correct_control" if gold == predicted else "bidirectional_confusion"
            listening_rows.append({
                "pair_id": pair_id,
                "pair_labels": [labels[left], labels[right]],
                "example_id": f"pair{pair_id:02d}_{example_index:02d}",
                "kind": kind,
                "gold_label": labels[gold],
                "clean_predicted_label": labels[predicted],
                "mixture_predicted_label": labels[int(mixture_prediction[index])],
                "component_path": str(event["component_path"]),
                "mixture_path": str(event["mixture_path"]),
                "onset_seconds": float(event["onset_seconds"]),
                "offset_seconds": float(event["offset_seconds"]),
                "scene_id": str(event["scene_id"]),
                "event_id": str(event.get("event_id") or ""),
                "source_id": str(event.get("source_id") or ""),
                "source_video_id": str(event.get("source_video_id") or ""),
            })
    listening_manifest = output_dir / "listening_manifest.jsonl"
    _atomic_write_text(
        listening_manifest,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in listening_rows),
    )
    group_miss = mixture["predicted_groups"].ne(true_group)
    routed_correct = mixture["routed_order"][:, 0].eq(targets)
    decomposition = {
        "correct_fraction_↑": float(routed_correct.float().mean()),
        "router_group_miss_fraction_↓": float(group_miss.float().mean()),
        "within_group_leaf_miss_fraction_↓": float((~group_miss & ~routed_correct).float().mean()),
        "sums_to_one": float(routed_correct.float().mean() + group_miss.float().mean() + (~group_miss & ~routed_correct).float().mean()),
    }
    future_eval = future_eval_availability(args.accepted_index.resolve(), args.matched_manifest.resolve(), labels)
    class_csv = output_dir / "class_failure_audit.csv"
    pair_csv = output_dir / "confusion_pair_candidates.csv"
    write_csv(class_rows, class_csv); write_csv(pair_rows, pair_csv)
    proposal = {
        "format": FORMAT + "_proposal",
        "design_source": "V5 design-dev only; matched predictions not read",
        "automatic_merge_applied": False,
        "rule": "manual listening is required; high-confidence candidates need bidirectional clean confusion >=0.15 combined and the same immediate AudioSet parent",
        "alias_clusters_for_listening_review": [
            {
                "cluster_id": index,
                "label_ids": component,
                "labels": [labels[value] for value in component],
            }
            for index, component in enumerate(nontrivial)
        ],
        "collapsed_metric_simulation_not_a_trained_result": collapsed,
    }
    proposal_path = output_dir / "hierarchy_proposal.json"
    _atomic_json(proposal, proposal_path)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "design-dev hierarchy/data audit; no training; no matched prediction used",
        "checkpoint_frozen": True,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "dev": {
            "mixture": stripped_eval(mixture),
            "clean_components": stripped_eval(clean),
            "error_decomposition": decomposition,
        },
        "audit_counts": {
            "classes": len(labels),
            "candidate_pairs": len(pair_rows),
            "high_confidence_merge_alias_pairs": len(high_confidence_edges),
            "listening_examples": len(listening_rows),
            "classes_clean_top1_below_0_50": sum(float(row["clean_oracle_top1_↑"]) < 0.50 for row in class_rows),
            "classes_clean_mixture_gap_ge_0_15": sum(float(row["clean_minus_mixture_gap_↓"]) >= 0.15 for row in class_rows),
            "classes_router_group_accuracy_below_0_70": sum(float(row["router_group_accuracy_↑"]) < 0.70 for row in class_rows),
        },
        "proposal": proposal,
        "future_eval_availability": future_eval,
        "decision": {
            "next_model_training_authorized": False,
            "required_next_action": "listen_and_adjudicate_high_confidence_alias_candidates_then_acquire_missing_disjoint_eval_sources",
        },
        "artifacts": {
            "class_failure_audit_csv": str(class_csv),
            "class_failure_audit_sha256": _sha256_file(class_csv),
            "confusion_pair_candidates_csv": str(pair_csv),
            "confusion_pair_candidates_sha256": _sha256_file(pair_csv),
            "hierarchy_proposal": str(proposal_path),
            "hierarchy_proposal_sha256": _sha256_file(proposal_path),
            "listening_manifest": str(listening_manifest),
            "listening_manifest_sha256": _sha256_file(listening_manifest),
            "dev_manifest_sha256": _sha256_file(args.dev_manifest.resolve()),
            "matched_manifest_used_only_for_identity_blocking_sha256": _sha256_file(args.matched_manifest.resolve()),
            "official_ontology_sha256": _sha256_file(args.ontology_json.resolve()),
        },
    }
    receipt_path = output_dir / "receipt.json"
    _atomic_json(receipt, receipt_path)
    print(json.dumps({
        "complete": True,
        "dev": receipt["dev"],
        "audit_counts": receipt["audit_counts"],
        "collapsed": collapsed,
        "future_eval": future_eval,
        "decision": receipt["decision"],
        "receipt": str(receipt_path),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
