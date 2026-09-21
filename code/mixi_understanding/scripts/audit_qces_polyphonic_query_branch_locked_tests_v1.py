#!/usr/bin/env python3
"""Locked-test audit for the polyphonic QCES interval-query branch.

The script compares the original relational-event-slots checkpoint with the
query-only fine-tuned checkpoint.  It never selects K or a threshold on test:
the fixed curve K in {1, 2, 3, 4, 6, 8} is reported in full.  K=8 is a proposal
ceiling, not a deployable decoder result.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from torch.utils.data import DataLoader

from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
)
from mixi_understanding.scripts.audit_qces_relational_raw_queries_v1 import (
    TOP_K_VALUES,
    _audit_k,
)
from mixi_understanding.scripts.train_qces_polyphonic_query_branch_v1 import (
    POLYPHONIC_CHECKPOINT_FORMAT,
    _state_hash,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import load_bound_qa_manifest_v2
from mixi_understanding.scripts.train_qces_relational_event_slots_v1 import (
    CHECKPOINT_FORMAT,
    SceneSlotDataset,
    collate_scene_slots,
    collect_predictions,
)


FORMAT = "qces_polyphonic_query_branch_locked_test_audit_v1"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dense_index: Path
    scene_list: Path
    qa_manifest: Path
    historical_hysteresis_receipt: Path


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    nonoverlap = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/multievent"
    repeated = PROJECT_ROOT / "outputs/qces_full191_repeated_ordinal_stress_test_v1"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_onset_stress_test_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-checkpoint", type=Path,
        default=base / "relational_event_slots_v1/relational_event_slots_v1_best.pt",
    )
    parser.add_argument(
        "--tuned-checkpoint", type=Path,
        default=base / "polyphonic_query_branch_v1/polyphonic_query_branch_best.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=base / "polyphonic_query_branch_v1_locked_tests/receipt.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(
        datasets=(
            DatasetSpec(
                "nonoverlap_locked_test",
                base / "dense_multi_test_v2/index.json",
                nonoverlap / "scene_ids_test.txt",
                nonoverlap / "qa_manifest_test.jsonl",
                base / "relational_event_slots_v1_locked_test/receipt.json",
            ),
            DatasetSpec(
                "repeated_ordinal_stress",
                base / "dense_repeated_ordinal_stress_test_v2/index.json",
                repeated / "scene_ids_test.txt",
                repeated / "qa_manifest_test.jsonl",
                base / "relational_event_slots_v1_repeat_stress_eval/receipt.json",
            ),
            DatasetSpec(
                "overlap_onset_stress",
                base / "dense_overlap_onset_stress_test_v1/index.json",
                overlap / "scene_ids_test.txt",
                overlap / "qa_manifest_test.jsonl",
                base / "relational_event_slots_v1_overlap_onset_stress_eval/receipt.json",
            ),
        )
    )
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_model(path: Path, device: torch.device) -> tuple[dict[str, Any], RelationalEventSlotsV1]:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("format") not in {CHECKPOINT_FORMAT, POLYPHONIC_CHECKPOINT_FORMAT}:
        raise ValueError(f"unsupported checkpoint format: {checkpoint.get('format')}")
    model = RelationalEventSlotsV1(RelationalEventSlotsV1Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return checkpoint, model


def _run_dataset(
    spec: DatasetSpec,
    *,
    models: dict[str, RelationalEventSlotsV1],
    device: torch.device,
    batch_size: int,
    cases_handle: Any,
) -> dict[str, Any]:
    store = DenseFeatureStore([spec.dense_index.resolve()], cache_size=8)
    scene_ids = load_scene_list(spec.scene_list.resolve())
    bound_qa = load_bound_qa_manifest_v2(
        spec.qa_manifest.resolve(), allowed_scene_ids=scene_ids, store=store, max_ordinal=10
    )
    dataset = SceneSlotDataset(store, scene_ids, preload=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_scene_slots,
    )
    model_results: dict[str, Any] = {}
    for model_name, model in models.items():
        predictions = collect_predictions(model, loader, device)
        curve = []
        for k in TOP_K_VALUES:
            metrics, rows = _audit_k(predictions, bound_qa, k=k)
            curve.append(metrics)
            if k == 8:
                for row in rows:
                    cases_handle.write(
                        json.dumps(
                            {"dataset": spec.name, "model": model_name, **row},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
        model_results[model_name] = {
            "fixed_top_k_curve": curve,
            "oracle_all_8_query_ceiling": curve[-1],
        }
    historical = _json(spec.historical_hysteresis_receipt)
    return {
        "scene_count": len(scene_ids),
        "qa_count": len(bound_qa),
        "dense_index": str(spec.dense_index.resolve()),
        "dense_index_sha256": _sha256_file(spec.dense_index.resolve()),
        "scene_list_sha256": _sha256_file(spec.scene_list.resolve()),
        "qa_manifest_sha256": _sha256_file(spec.qa_manifest.resolve()),
        "historical_hysteresis": {
            "checkpoint_sha256": historical["checkpoint_sha256"],
            "metrics": historical["metrics"],
            "unchanged_by_query_tuning": True,
        },
        "models": model_results,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    original_path = args.original_checkpoint.resolve()
    tuned_path = args.tuned_checkpoint.resolve()
    device = _device(args.device)
    original, original_model = _load_model(original_path, device)
    tuned, tuned_model = _load_model(tuned_path, device)
    if original.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("original checkpoint has wrong format")
    if tuned.get("format") != POLYPHONIC_CHECKPOINT_FORMAT:
        raise ValueError("tuned checkpoint has wrong format")
    if tuned["initial_checkpoint_sha256"] != _sha256_file(original_path):
        raise ValueError("tuned checkpoint was not initialized from the requested original")
    frozen_names = list(tuned["frozen_parameter_names"])
    original_frozen_hash = _state_hash(original["model_state_dict"], frozen_names)
    tuned_frozen_hash = _state_hash(tuned["model_state_dict"], frozen_names)
    if original_frozen_hash != tuned_frozen_hash:
        raise RuntimeError("frozen encoder/frame-head state differs after query tuning")

    cases_path = output.parent / "oracle_k8_cases.jsonl"
    datasets: dict[str, Any] = {}
    with cases_path.open("w", encoding="utf-8") as cases_handle:
        for spec in args.datasets:
            datasets[spec.name] = _run_dataset(
                spec,
                models={"original_query_branch": original_model, "tuned_query_branch": tuned_model},
                device=device,
                batch_size=args.batch_size,
                cases_handle=cases_handle,
            )

    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_threshold_or_k_tuning": False,
        "answer_label_used_for_query_generation_or_ranking": False,
        "protocol": {
            "fixed_top_k_values": list(TOP_K_VALUES),
            "ranking": "DETR query objectness only",
            "k8_interpretation": "oracle proposal ceiling, not deployment accuracy",
        },
        "checkpoints": {
            "original": {
                "path": str(original_path),
                "sha256": _sha256_file(original_path),
                "epoch": int(original["epoch"]),
            },
            "tuned": {
                "path": str(tuned_path),
                "sha256": _sha256_file(tuned_path),
                "epoch": int(tuned["epoch"]),
            },
        },
        "frozen_branch_audit": {
            "parameter_count": len(frozen_names),
            "original_sha256": original_frozen_hash,
            "tuned_sha256": tuned_frozen_hash,
            "exact_match": True,
        },
        "datasets": datasets,
        "oracle_k8_cases": {"path": str(cases_path), "sha256": _sha256_file(cases_path)},
    }
    _atomic_json(receipt, output)
    compact = {
        name: {
            model_name: result["oracle_all_8_query_ceiling"]
            for model_name, result in dataset["models"].items()
        }
        for name, dataset in datasets.items()
    }
    print(json.dumps({"complete": True, "output": str(output), "k8": compact}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
