#!/usr/bin/env python3
"""Export static development predictions for the mentor Streamlit demo."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from mixi_understanding.qces.class_aware_red_epn_v1 import (
    ClassAwareRedEpnV1,
    ClassAwareRedEpnV1Config,
    interval_iou,
)
from mixi_understanding.scripts.train_qces_gold_natural_v3_acoustic_pair_ranker_v2 import (
    AcousticPairNoneRanker,
    AcousticRankerConfig,
    DEFAULT_BASE,
    DEFAULT_DATA,
    QuestionDataset,
    build_question_examples,
    collect_acoustic_scenes,
    pad_scene_bank,
    to_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _device,
    _sha256_file,
    load_scene_list,
)


DEFAULT_RUN = DEFAULT_BASE / "question_conditioned_acoustic_pair_ranker_v2"
DEFAULT_OUTPUT = (
    DEFAULT_BASE / "question_conditioned_acoustic_pair_ranker_v2_demo"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranker-checkpoint",
        type=Path,
        default=DEFAULT_RUN / "acoustic_pair_ranker_v2_best.pt",
    )
    parser.add_argument(
        "--ranker-receipt", type=Path, default=DEFAULT_RUN / "receipt.json"
    )
    parser.add_argument(
        "--proposal-checkpoint",
        type=Path,
        default=DEFAULT_BASE
        / "overlap_aware_red_epn_v2/overlap_aware_red_epn_v2_best.pt",
    )
    parser.add_argument(
        "--dev-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_dev/index.json",
    )
    parser.add_argument(
        "--dev-scenes", type=Path, default=DEFAULT_DATA / "scene_ids_overlap_dev.txt"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--proposal-batch-size", type=int, default=16)
    parser.add_argument("--rank-batch-size", type=int, default=16)
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser.parse_args()


def display_label(value: str) -> str:
    return value.replace("_and_", " & ").replace("_", " ").strip()


def _event_iou(
    predicted_start: float,
    predicted_end: float,
    gold_start: float,
    gold_end: float,
) -> float:
    return interval_iou(
        (predicted_start, predicted_end), (gold_start, gold_end)
    )


def classify_case(
    *,
    no_evidence: bool,
    predicted_none: bool,
    anchor_correct: bool,
    label_correct: bool,
    answer_correct: bool,
) -> str:
    if no_evidence:
        if predicted_none and anchor_correct:
            return "correct_no_evidence"
        if predicted_none:
            return "unverified_no_evidence"
        return "wrong_no_evidence"
    if predicted_none:
        return "wrong_abstention"
    if answer_correct and anchor_correct:
        return "correct_joint_evidence"
    if answer_correct:
        return "correct_answer_wrong_anchor"
    if label_correct:
        return "correct_label_wrong_boundary"
    return "wrong_answer"


def _component_for_label(
    events: list[Mapping[str, Any]], label: str
) -> str | None:
    for event in events:
        if str(event.get("label")) == label:
            path = event.get("component_path")
            return None if path is None else str(Path(str(path)).resolve())
    return None


def _write_jsonl_atomic(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    checkpoint_path = args.ranker_checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = AcousticRankerConfig(**dict(checkpoint["config"]))
    proposal_payload = torch.load(
        args.proposal_checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    proposal_config = ClassAwareRedEpnV1Config(**dict(proposal_payload["config"]))
    proposal_model = ClassAwareRedEpnV1(proposal_config)
    proposal_model.load_state_dict(proposal_payload["model_state_dict"], strict=True)
    proposal_model.to(device).eval()
    store = DenseFeatureStore([args.dev_index.resolve()], cache_size=16)
    scene_ids = load_scene_list(args.dev_scenes.resolve())
    scenes = collect_acoustic_scenes(
        proposal_model,
        store,
        scene_ids,
        device=device,
        batch_size=args.proposal_batch_size,
        max_proposals=config.max_proposals,
        max_scenes=args.max_scenes,
        iou_threshold=0.30,
    )
    del proposal_model
    torch.cuda.empty_cache()
    banks = [pad_scene_bank(scene, config) for scene in scenes]
    examples, counts = build_question_examples(scenes, config, iou_threshold=0.30)
    dataset = QuestionDataset(banks, examples)
    loader = DataLoader(
        dataset,
        batch_size=args.rank_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = AcousticPairNoneRanker(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    none_bias = float(checkpoint["none_bias"])
    labels = list(store.labels or [])
    output_rows: list[dict[str, Any]] = []
    global_index = 0
    with torch.inference_mode():
        for raw in loader:
            batch = to_device(raw, device)
            scores, _ = model(batch)
            scores = scores + batch["candidate_is_none"].to(scores.dtype) * none_bias
            top2 = scores.topk(k=2, dim=1)
            for local_index in range(scores.shape[0]):
                example = examples[global_index]
                scene_index = int(example["scene_index"])
                scene = scenes[scene_index]
                scene_id = str(scene["scene_id"])
                metadata = store.metadata(scene_id)
                duration = float(metadata.get("duration_seconds", 10.0))
                choice = int(top2.indices[local_index, 0])
                prediction_available = bool(raw["candidate_mask"][local_index, choice])
                score_margin = float(
                    top2.values[local_index, 0] - top2.values[local_index, 1]
                )
                predicted_none = bool(
                    prediction_available
                    and raw["candidate_is_none"][local_index, choice]
                )
                no_evidence = bool(raw["no_evidence"][local_index])
                predicted_anchor_start = float(
                    raw["candidate_anchor_start"][local_index, choice]
                )
                predicted_anchor_end = float(
                    raw["candidate_anchor_end"][local_index, choice]
                )
                gold_anchor_start = float(raw["gold_anchor_start"][local_index])
                gold_anchor_end = float(raw["gold_anchor_end"][local_index])
                anchor_iou = (
                    _event_iou(
                        predicted_anchor_start,
                        predicted_anchor_end,
                        gold_anchor_start,
                        gold_anchor_end,
                    )
                    if prediction_available
                    else 0.0
                )
                anchor_correct = anchor_iou >= 0.30
                gold_answer_id = int(raw["gold_answer_label"][local_index])
                predicted_answer_id = int(
                    raw["candidate_answer_label"][local_index, choice]
                )
                gold_answer_start = float(raw["gold_answer_start"][local_index])
                gold_answer_end = float(raw["gold_answer_end"][local_index])
                predicted_answer_start = float(
                    raw["candidate_answer_start"][local_index, choice]
                )
                predicted_answer_end = float(
                    raw["candidate_answer_end"][local_index, choice]
                )
                label_correct = bool(
                    prediction_available
                    and
                    not no_evidence
                    and not predicted_none
                    and predicted_answer_id == gold_answer_id
                )
                answer_iou = (
                    0.0
                    if no_evidence or predicted_none
                    else _event_iou(
                        predicted_answer_start,
                        predicted_answer_end,
                        gold_answer_start,
                        gold_answer_end,
                    )
                )
                answer_correct = label_correct and answer_iou >= 0.30
                category = (
                    classify_case(
                        no_evidence=no_evidence,
                        predicted_none=predicted_none,
                        anchor_correct=anchor_correct,
                        label_correct=label_correct,
                        answer_correct=answer_correct,
                    )
                    if prediction_available
                    else "missing_anchor_proposal"
                )
                anchor_label_id = int(raw["anchor_label_id"][local_index])
                relation_id = int(raw["relation_id"][local_index])
                anchor_label = labels[anchor_label_id]
                gold_answer_label = None if no_evidence else labels[gold_answer_id]
                predicted_answer_label = (
                    None
                    if predicted_none or not prediction_available
                    else labels[predicted_answer_id]
                )
                relation = "before" if relation_id == 0 else "after"
                question = (
                    f"Which sound occurs immediately {relation} "
                    f"{display_label(anchor_label)}?"
                )
                metadata_events = [
                    event
                    for event in metadata.get("events", [])
                    if str(event.get("event_kind", "semantic")) == "semantic"
                ]
                scene_events = [
                    {
                        "label": str(event["label"]),
                        "display_label": display_label(str(event["label"])),
                        "onset_seconds": float(event["onset_seconds"]),
                        "offset_seconds": float(event["offset_seconds"]),
                        "component_path": (
                            None
                            if event.get("component_path") is None
                            else str(Path(str(event["component_path"])).resolve())
                        ),
                    }
                    for event in sorted(
                        metadata_events,
                        key=lambda item: (
                            float(item["onset_seconds"]),
                            float(item["offset_seconds"]),
                        ),
                    )
                ]
                if not prediction_available:
                    predicted_window = None
                elif predicted_none:
                    predicted_window = (
                        [0.0, predicted_anchor_end * duration]
                        if relation_id == 0
                        else [predicted_anchor_start * duration, duration]
                    )
                else:
                    predicted_window = [
                        min(predicted_anchor_start, predicted_answer_start) * duration,
                        max(predicted_anchor_end, predicted_answer_end) * duration,
                    ]
                if no_evidence:
                    oracle_window = (
                        [0.0, gold_anchor_end * duration]
                        if relation_id == 0
                        else [gold_anchor_start * duration, duration]
                    )
                else:
                    oracle_window = [
                        min(gold_anchor_start, gold_answer_start) * duration,
                        max(gold_anchor_end, gold_answer_end) * duration,
                    ]
                output_rows.append(
                    {
                        "record_id": f"{scene_id}:q{global_index:05d}",
                        "scene_id": scene_id,
                        "category": category,
                        "question": question,
                        "relation": relation,
                        "duration_seconds": duration,
                        "mixture_path": str(Path(str(metadata["mixture_path"])).resolve()),
                        "scene_events": scene_events,
                        "anchor_label": anchor_label,
                        "anchor_display_label": display_label(anchor_label),
                        "gold_answer_label": gold_answer_label,
                        "gold_answer_display_label": (
                            "NONE"
                            if gold_answer_label is None
                            else display_label(gold_answer_label)
                        ),
                        "predicted_answer_label": predicted_answer_label,
                        "predicted_answer_display_label": (
                            "NO VALID PROPOSAL"
                            if not prediction_available
                            else "NONE"
                            if predicted_answer_label is None
                            else display_label(predicted_answer_label)
                        ),
                        "prediction_available": prediction_available,
                        "predicted_none": predicted_none,
                        "no_evidence": no_evidence,
                        "candidate_supported": bool(raw["target_available"][local_index]),
                        "anchor_iou": anchor_iou,
                        "answer_iou": answer_iou,
                        "anchor_correct": anchor_correct,
                        "label_correct": label_correct,
                        "answer_correct": answer_correct,
                        "joint_correct": bool(answer_correct and anchor_correct),
                        "score_margin": score_margin,
                        "predicted_anchor_seconds": [
                            predicted_anchor_start * duration,
                            predicted_anchor_end * duration,
                        ] if prediction_available else None,
                        "gold_anchor_seconds": [
                            gold_anchor_start * duration,
                            gold_anchor_end * duration,
                        ],
                        "predicted_answer_seconds": (
                            None
                            if predicted_none or not prediction_available
                            else [
                                predicted_answer_start * duration,
                                predicted_answer_end * duration,
                            ]
                        ),
                        "gold_answer_seconds": (
                            None
                            if no_evidence
                            else [
                                gold_answer_start * duration,
                                gold_answer_end * duration,
                            ]
                        ),
                        "predicted_evidence_window_seconds": predicted_window,
                        "oracle_evidence_window_seconds": oracle_window,
                        "gold_anchor_component_path": _component_for_label(
                            metadata_events, anchor_label
                        ),
                        "gold_answer_component_path": (
                            None
                            if gold_answer_label is None
                            else _component_for_label(
                                metadata_events, gold_answer_label
                            )
                        ),
                        "recipe_kind": str(metadata.get("recipe_kind", "unknown")),
                        "maximum_concurrency": int(
                            metadata.get("maximum_concurrency", 0)
                        ),
                    }
                )
                global_index += 1
            if global_index % 1024 < scores.shape[0]:
                print(f"rank inference: {global_index}/{len(dataset)} questions", flush=True)

    ranker_receipt = json.loads(
        args.ranker_receipt.resolve().read_text(encoding="utf-8")
    )
    v1_path = DEFAULT_BASE / "listwise_pair_none_ranker_v1/receipt.json"
    v1_receipt = (
        json.loads(v1_path.read_text(encoding="utf-8"))
        if v1_path.is_file()
        else None
    )
    proposal_path = DEFAULT_BASE / "overlap_aware_red_epn_v2/adjacent_pair_audit.json"
    proposal_audit = (
        json.loads(proposal_path.read_text(encoding="utf-8"))
        if proposal_path.is_file()
        else None
    )
    category_counts = Counter(row["category"] for row in output_rows)
    summary = {
        "format": "qces_acoustic_pair_ranker_mentor_demo_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": len(output_rows),
        "scenes": len(scenes),
        "classes": len(labels),
        "category_counts": dict(sorted(category_counts.items())),
        "question_counts": counts,
        "ranker_best": ranker_receipt["best_metrics"],
        "semantic_best": ranker_receipt["semantic_best_metrics"],
        "feature_only_v1_best": (
            None if v1_receipt is None else v1_receipt["best_metrics"]
        ),
        "proposal_audit": proposal_audit,
        "ranker_checkpoint": str(checkpoint_path),
        "ranker_checkpoint_sha256": _sha256_file(checkpoint_path),
        "cases_file": str((args.output_dir.resolve() / "cases.jsonl")),
    }
    output_dir = args.output_dir.resolve()
    _write_jsonl_atomic(output_rows, output_dir / "cases.jsonl")
    _atomic_json(summary, output_dir / "summary.json")
    print(
        json.dumps(
            {
                "complete": True,
                "cases": len(output_rows),
                "categories": summary["category_counts"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
