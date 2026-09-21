#!/usr/bin/env python3
"""Evaluate the trained semantic+span separator with oracle/predicted inputs.

The four-way matrix isolates semantic and temporal error propagation:

* oracle label + oracle span;
* oracle label + predicted span;
* predicted label + oracle span;
* predicted label + predicted span (deployable model path).

No AudioQA model is used: the system answer is the semantic label predicted
for the answer interval, or NONE when the temporal decoder abstains.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
)
from mixi_understanding.scripts.evaluate_qces_polyphonic_deployable_decoder_v1 import (
    FIXED_AUDIO_SECONDS,
    collect,
    decode,
    interval_iou,
)
from mixi_understanding.scripts.train_qces_onset_contrast_semantic_v1 import (
    CHECKPOINT_FORMAT as SEMANTIC_CHECKPOINT_FORMAT,
    Config as SemanticConfig,
    OnsetContrastSemanticHead,
    interval_components,
)
from mixi_understanding.scripts.train_qces_polyphonic_query_branch_v1 import (
    POLYPHONIC_CHECKPOINT_FORMAT,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_span_conditioned_separator_v1 import (
    CHECKPOINT_FORMAT as SEPARATOR_CHECKPOINT_FORMAT,
    RATE,
    SpanMaskNetwork,
    _separator_forward,
)


FORMAT = "qces_semantic_span_separator_integration_evaluation_v1"
MODES = (
    "oracle_label_oracle_span",
    "oracle_label_predicted_span",
    "predicted_label_oracle_span",
    "predicted_label_predicted_span",
)
SCENE_SAMPLES = int(FIXED_AUDIO_SECONDS * RATE)


@dataclass
class Question:
    item_id: str
    scene_id: str
    anchor_label: str
    anchor_ordinal: int
    relation: str
    no_evidence: bool
    gold_anchor: tuple[float, float]
    gold_answer: tuple[float, float] | None
    answer_label_diagnostic_only: str | None
    answer_event_id: str | None
    question: str
    predicted_none: bool = True
    predicted_span: tuple[float, float] | None = None
    predicted_label_id: int | None = None
    predicted_label: str | None = None
    predicted_label_confidence: float | None = None
    oracle_span_predicted_label_id: int | None = None
    oracle_span_predicted_label: str | None = None
    oracle_span_predicted_label_confidence: float | None = None
    anchor_iou: float = 0.0
    answer_iou: float = 0.0


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    components = PROJECT_ROOT / "outputs/qces_full191_overlap_components_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporal-checkpoint", type=Path,
        default=base / "polyphonic_query_branch_v2/polyphonic_query_branch_best.pt",
    )
    parser.add_argument(
        "--decoder-receipt", type=Path,
        default=base / "polyphonic_deployable_decoder_v1/receipt.json",
    )
    parser.add_argument(
        "--semantic-checkpoint", type=Path,
        default=base / "onset_contrast_semantic_v1/onset_contrast_semantic_best.pt",
    )
    parser.add_argument(
        "--separator-checkpoint", type=Path,
        default=PROJECT_ROOT / "outputs/qces_semantic_span_separator_full191_v1/best.pt",
    )
    parser.add_argument(
        "--label-embedding-cache", type=Path,
        default=PROJECT_ROOT / "outputs/qces_label_clap191_v1.pt",
    )
    parser.add_argument("--dense-index", type=Path, default=base / "dense_overlap_query_dev_v2/index.json")
    parser.add_argument("--scene-list", type=Path, default=data / "scene_ids_overlap_dev.txt")
    parser.add_argument(
        "--component-manifest", type=Path, default=components / "event_components_dev.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_semantic_span_separator_integration_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temporal-batch-size", type=int, default=64)
    parser.add_argument("--semantic-batch-size", type=int, default=128)
    parser.add_argument("--separator-batch-size", type=int, default=32)
    parser.add_argument("--max-questions", type=int, default=0)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _normalized_interval(event: Mapping[str, Any]) -> tuple[float, float]:
    return (
        float(event["onset_seconds"]) / FIXED_AUDIO_SECONDS,
        float(event["offset_seconds"]) / FIXED_AUDIO_SECONDS,
    )


def build_questions(store: DenseFeatureStore, scene_ids: Sequence[str]) -> list[Question]:
    rows: list[Question] = []
    for scene_id in scene_ids:
        events = sorted(
            store.metadata(scene_id)["gold_events"],
            key=lambda event: (
                float(event["onset_seconds"]), float(event["offset_seconds"]), str(event["event_id"])
            ),
        )
        for position, anchor in enumerate(events):
            for relation, delta in (("before", -1), ("after", 1)):
                answer_position = position + delta
                answer = events[answer_position] if 0 <= answer_position < len(events) else None
                rows.append(
                    Question(
                        item_id=f"{scene_id}:{relation}:{position}",
                        scene_id=scene_id,
                        anchor_label=str(anchor["label"]),
                        anchor_ordinal=1,
                        relation=relation,
                        no_evidence=answer is None,
                        gold_anchor=_normalized_interval(anchor),
                        gold_answer=None if answer is None else _normalized_interval(answer),
                        answer_label_diagnostic_only=None if answer is None else str(answer["label"]),
                        answer_event_id=None if answer is None else str(answer["event_id"]),
                        question=(
                            f"What sound has the immediate {relation} onset relative to "
                            f"{anchor['label']}?"
                        ),
                    )
                )
    return rows


def attach_temporal_predictions(
    questions: Sequence[Question], predictions: Mapping[str, Mapping[str, Any]],
    label_to_id: Mapping[str, int], threshold: float,
) -> None:
    for question in questions:
        result = decode(predictions[question.scene_id], question, label_to_id, threshold)
        anchor = result["anchor"]
        answer = result["answer"]
        question.predicted_none = bool(result["predicted_none"])
        question.predicted_span = (
            None if answer is None else (float(answer["start"]), float(answer["end"]))
        )
        anchor_span = None if anchor is None else (float(anchor["start"]), float(anchor["end"]))
        question.anchor_iou = interval_iou(anchor_span, question.gold_anchor)
        question.answer_iou = interval_iou(question.predicted_span, question.gold_answer)


@torch.inference_mode()
def attach_semantic_predictions(
    questions: Sequence[Question], store: DenseFeatureStore,
    model: OnsetContrastSemanticHead, labels: Sequence[str], device: torch.device,
    batch_size: int, *, span_source: str,
) -> None:
    if span_source not in {"predicted", "oracle"}:
        raise ValueError(f"invalid semantic span source: {span_source}")
    selected = [
        question for question in questions
        if (question.predicted_span if span_source == "predicted" else question.gold_answer) is not None
    ]
    model.eval()
    for begin in range(0, len(selected), batch_size):
        batch_questions = selected[begin: begin + batch_size]
        dense_rows = [store.get(question.scene_id) for question in batch_questions]
        features = torch.stack([row["features"].float() for row in dense_rows]).to(device)
        detector_logits = torch.stack([row["logits"].float() for row in dense_rows]).to(device)
        valid = torch.tensor([int(row["valid_frames"]) for row in dense_rows], device=device)
        starts = []
        ends = []
        for question, maximum in zip(batch_questions, valid.tolist(), strict=True):
            interval = question.predicted_span if span_source == "predicted" else question.gold_answer
            assert interval is not None
            start = max(0, min(int(maximum) - 1, int(math.floor(interval[0] * 250))))
            end = max(start + 1, min(int(maximum), int(math.ceil(interval[1] * 250))))
            starts.append(start)
            ends.append(end)
        acoustic, detector = interval_components(
            features, detector_logits,
            torch.tensor(starts, device=device), torch.tensor(ends, device=device), valid,
            context_frames=model.config.context_frames, early_frames=model.config.early_frames,
        )
        probabilities = torch.softmax(model(acoustic, detector), dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        for question, label_id, score in zip(
            batch_questions, predicted.cpu().tolist(), confidence.cpu().tolist(), strict=True
        ):
            if span_source == "predicted":
                question.predicted_label_id = int(label_id)
                question.predicted_label = str(labels[label_id])
                question.predicted_label_confidence = float(score)
            else:
                question.oracle_span_predicted_label_id = int(label_id)
                question.oracle_span_predicted_label = str(labels[label_id])
                question.oracle_span_predicted_label_confidence = float(score)


def _load_wave(path: str | Path) -> torch.Tensor:
    value, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(sample_rate) != RATE:
        raise RuntimeError(f"{path}: expected {RATE} Hz, got {sample_rate}")
    return torch.from_numpy(value.mean(axis=1)).float()


@lru_cache(maxsize=96)
def _cached_mixture(path: str) -> torch.Tensor:
    wave = _load_wave(path)
    if wave.numel() != SCENE_SAMPLES:
        raise RuntimeError(f"{path}: expected {SCENE_SAMPLES} samples, got {wave.numel()}")
    return wave


@lru_cache(maxsize=768)
def _cached_component(path: str) -> torch.Tensor:
    return _load_wave(path)


def _summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _mode_condition(
    mode: str, question: Question, label_to_id: Mapping[str, int],
) -> tuple[int, tuple[float, float]] | None:
    if question.no_evidence and "oracle_span" in mode:
        return None
    span = question.gold_answer if mode.endswith("oracle_span") else question.predicted_span
    if span is None:
        return None
    if mode.startswith("oracle_label"):
        if question.answer_label_diagnostic_only is None:
            return None
        label_id = label_to_id[question.answer_label_diagnostic_only]
    else:
        label_id = (
            question.oracle_span_predicted_label_id
            if mode.endswith("oracle_span") else question.predicted_label_id
        )
        if label_id is None:
            return None
    return int(label_id), span


def _prepare_chunk(
    mixture: torch.Tensor, normalized_span: tuple[float, float], chunk_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, int]]:
    onset = max(0, min(SCENE_SAMPLES - 1, int(round(normalized_span[0] * SCENE_SAMPLES))))
    offset = max(onset + 1, min(SCENE_SAMPLES, int(round(normalized_span[1] * SCENE_SAMPLES))))
    center = (onset + offset) // 2
    chunk_start = max(0, min(SCENE_SAMPLES - chunk_samples, center - chunk_samples // 2))
    chunk = mixture[chunk_start: chunk_start + chunk_samples]
    span = torch.zeros(chunk_samples, dtype=torch.float32)
    local_start = max(0, onset - chunk_start)
    local_end = min(chunk_samples, offset - chunk_start)
    if local_end > local_start:
        span[local_start:local_end] = 1.0
    return chunk, span, chunk_start, (onset, offset)


def _target_wave(question: Question, components: Mapping[str, Mapping[str, Any]]) -> torch.Tensor:
    target = torch.zeros(SCENE_SAMPLES, dtype=torch.float32)
    if question.answer_event_id is None:
        return target
    row = components[question.answer_event_id]
    component = _cached_component(str(row["component_path"]))
    onset = int(row["onset_sample"])
    target[onset: onset + component.numel()] = component
    return target


@torch.inference_mode()
def evaluate_mode(
    *, mode: str, questions: Sequence[Question], components: Mapping[str, Mapping[str, Any]],
    scene_mixtures: Mapping[str, str], separator: SpanMaskNetwork,
    label_to_id: Mapping[str, int], chunk_samples: int, device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    separator.eval()
    predicted_sd: list[float] = []
    baseline_sd: list[float] = []
    predicted_si: list[float] = []
    baseline_si: list[float] = []
    positive_cases: list[dict[str, Any]] = []
    no_evidence_retained: list[float] = []

    for batch_begin in range(0, len(questions), batch_size):
        batch_questions = questions[batch_begin: batch_begin + batch_size]
        batch_count = len(batch_questions)
        mixtures = [_cached_mixture(scene_mixtures[q.scene_id]) for q in batch_questions]
        targets = torch.stack([_target_wave(q, components) for q in batch_questions])
        baseline = torch.zeros(batch_count, SCENE_SAMPLES, dtype=torch.float32)
        predicted = torch.zeros_like(baseline)
        valid_indices: list[int] = []
        chunks: list[torch.Tensor] = []
        spans: list[torch.Tensor] = []
        chunk_starts: list[int] = []
        label_ids: list[int] = []
        conditions: list[tuple[int, tuple[float, float]] | None] = []
        for index, (question, mixture) in enumerate(zip(batch_questions, mixtures, strict=True)):
            condition = _mode_condition(mode, question, label_to_id)
            conditions.append(condition)
            if condition is None:
                continue
            label_id, normalized_span = condition
            chunk, sample_span, chunk_start, full_span = _prepare_chunk(
                mixture, normalized_span, chunk_samples
            )
            baseline[index, full_span[0]:full_span[1]] = mixture[full_span[0]:full_span[1]]
            valid_indices.append(index)
            chunks.append(chunk)
            spans.append(sample_span)
            chunk_starts.append(chunk_start)
            label_ids.append(label_id)
        if valid_indices:
            chunk_batch = torch.stack(chunks).to(device, non_blocking=True)
            span_batch = torch.stack(spans).to(device, non_blocking=True)
            label_batch = torch.tensor(label_ids, dtype=torch.long, device=device)
            separated, _, _, _ = _separator_forward(separator, chunk_batch, span_batch, label_batch)
            separated = separated.cpu()
            for local, global_index in enumerate(valid_indices):
                start = chunk_starts[local]
                predicted[global_index, start:start + chunk_samples] = separated[local]

        positive_indices = [index for index, q in enumerate(batch_questions) if not q.no_evidence]
        if positive_indices:
            p = predicted[positive_indices].to(device)
            b = baseline[positive_indices].to(device)
            t = targets[positive_indices].to(device)
            p_sd = scale_dependent_sdr(p, t).cpu().tolist()
            b_sd = scale_dependent_sdr(b, t).cpu().tolist()
            p_si = scale_invariant_sdr(p, t).cpu().tolist()
            b_si = scale_invariant_sdr(b, t).cpu().tolist()
            predicted_sd.extend(p_sd)
            baseline_sd.extend(b_sd)
            predicted_si.extend(p_si)
            baseline_si.extend(b_si)
            for position, index in enumerate(positive_indices):
                question = batch_questions[index]
                positive_cases.append(
                    {
                        "item_id": question.item_id,
                        "mode": mode,
                        "sd_sdr_db": p_sd[position],
                        "baseline_sd_sdr_db": b_sd[position],
                        "sd_sdri_db": p_sd[position] - b_sd[position],
                    }
                )
        negative_indices = [index for index, q in enumerate(batch_questions) if q.no_evidence]
        for index in negative_indices:
            ratio = float(
                predicted[index].square().sum()
                / torch.as_tensor(mixtures[index]).square().sum().clamp_min(1e-8)
            )
            no_evidence_retained.append(ratio)
        if (batch_begin + batch_count) % (batch_size * 20) == 0:
            print(
                json.dumps(
                    {"mode": mode, "done": batch_begin + batch_count, "total": len(questions)},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    sd_gain = [p - b for p, b in zip(predicted_sd, baseline_sd, strict=True)]
    si_gain = [p - b for p, b in zip(predicted_si, baseline_si, strict=True)]
    positives = [q for q in questions if not q.no_evidence]
    negatives = [q for q in questions if q.no_evidence]
    label_predictions_used = mode.startswith("predicted_label")
    span_predictions_used = mode.endswith("predicted_span")
    if not label_predictions_used and not span_predictions_used:
        answerable_label_correct = len(positives)
    elif not label_predictions_used and span_predictions_used:
        answerable_label_correct = sum(not q.predicted_none for q in positives)
    elif label_predictions_used and not span_predictions_used:
        answerable_label_correct = sum(
            q.oracle_span_predicted_label == q.answer_label_diagnostic_only for q in positives
        )
    else:
        answerable_label_correct = sum(
            (not q.predicted_none) and q.predicted_label == q.answer_label_diagnostic_only
            for q in positives
        )
    no_evidence_correct = sum(q.predicted_none for q in negatives) if span_predictions_used else len(negatives)
    overall_correct = answerable_label_correct + no_evidence_correct
    answer_iou30 = sum(q.answer_iou >= 0.30 for q in positives) / max(len(positives), 1)
    answer_iou50 = sum(q.answer_iou >= 0.50 for q in positives) / max(len(positives), 1)
    metrics = {
        "mode": mode,
        "questions": len(questions),
        "answerable_questions": len(positives),
        "no_evidence_questions": len(negatives),
        "uses_oracle_label": not label_predictions_used,
        "uses_oracle_span": not span_predictions_used,
        "answerable_label_accuracy_↑": answerable_label_correct / max(len(positives), 1),
        "no_evidence_accuracy_↑": no_evidence_correct / max(len(negatives), 1),
        "overall_answer_accuracy_↑": overall_correct / max(len(questions), 1),
        "balanced_answer_no_evidence_accuracy_↑": 0.5 * (
            answerable_label_correct / max(len(positives), 1)
            + no_evidence_correct / max(len(negatives), 1)
        ),
        "answer_event_iou30_↑": 1.0 if not span_predictions_used else answer_iou30,
        "answer_event_iou50_↑": 1.0 if not span_predictions_used else answer_iou50,
        "evidence_sd_sdr_db_↑": _summary(predicted_sd),
        "window_baseline_sd_sdr_db_↑": _summary(baseline_sd),
        "evidence_sd_sdri_over_matching_window_db_↑": _summary(sd_gain),
        "evidence_sd_sdri_positive_rate_↑": float(np.mean(np.asarray(sd_gain) > 0.0)),
        "evidence_si_sdri_over_matching_window_db_↑": _summary(si_gain),
        "no_evidence_retained_energy_ratio_↓": float(np.mean(no_evidence_retained)),
        "oracle_best_of_window_separator_sd_sdr_db_↑": _summary(
            [max(p, b) for p, b in zip(predicted_sd, baseline_sd, strict=True)]
        ),
    }
    return metrics, positive_cases


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    store = DenseFeatureStore([args.dense_index.resolve()], cache_size=64)
    labels = list(store.labels or [])
    label_to_id = {label: index for index, label in enumerate(labels)}
    scene_ids = load_scene_list(args.scene_list.resolve())

    temporal_saved = torch.load(args.temporal_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if temporal_saved.get("format") != POLYPHONIC_CHECKPOINT_FORMAT:
        raise ValueError("invalid temporal checkpoint format")
    if list(temporal_saved["labels"]) != labels:
        raise ValueError("temporal checkpoint label order mismatch")
    temporal = RelationalEventSlotsV1(RelationalEventSlotsV1Config(**temporal_saved["config"])).to(device)
    temporal.load_state_dict(temporal_saved["model_state_dict"], strict=True)
    temporal.eval()
    temporal_predictions = collect(temporal, store, scene_ids, device, args.temporal_batch_size)
    decoder_receipt = json.loads(args.decoder_receipt.resolve().read_text(encoding="utf-8"))
    threshold = float(decoder_receipt["dev_split"]["locked_threshold"])
    questions = build_questions(store, scene_ids)
    if args.max_questions > 0:
        questions = questions[: args.max_questions]
    attach_temporal_predictions(questions, temporal_predictions, label_to_id, threshold)
    del temporal
    torch.cuda.empty_cache()

    semantic_saved = torch.load(args.semantic_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if semantic_saved.get("format") != SEMANTIC_CHECKPOINT_FORMAT:
        raise ValueError("invalid semantic checkpoint format")
    if list(semantic_saved["labels"]) != labels:
        raise ValueError("semantic checkpoint label order mismatch")
    semantic = OnsetContrastSemanticHead(SemanticConfig(**semantic_saved["config"])).to(device)
    semantic.load_state_dict(semantic_saved["model_state_dict"], strict=True)
    attach_semantic_predictions(
        questions, store, semantic, labels, device, args.semantic_batch_size,
        span_source="predicted",
    )
    attach_semantic_predictions(
        questions, store, semantic, labels, device, args.semantic_batch_size,
        span_source="oracle",
    )
    del semantic, temporal_predictions
    torch.cuda.empty_cache()

    component_rows = _read_jsonl(args.component_manifest.resolve())
    components = {str(row["event_id"]): row for row in component_rows}
    missing = sorted(
        q.answer_event_id for q in questions
        if q.answer_event_id is not None and q.answer_event_id not in components
    )
    if missing:
        raise RuntimeError(f"component manifest misses answer events: {missing[:5]}")
    scene_mixtures: dict[str, str] = {}
    for row in component_rows:
        scene_mixtures.setdefault(str(row["scene_id"]), str(row["mixture_path"]))

    label_cache = torch.load(args.label_embedding_cache.resolve(), map_location="cpu", weights_only=True)
    if list(label_cache["labels"]) != labels:
        raise ValueError("label CLAP cache order mismatch")
    separator_saved = torch.load(args.separator_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if separator_saved.get("format") != SEPARATOR_CHECKPOINT_FORMAT:
        raise ValueError("invalid separator checkpoint format")
    separator_config = separator_saved["config"]
    separator = SpanMaskNetwork(
        base_channels=int(separator_config["base_channels"]),
        semantic_channels=int(separator_config["semantic_channels"]),
        label_embeddings=label_cache["embeddings"],
    ).to(device)
    separator.load_state_dict(separator_saved["model_state"], strict=True)
    chunk_samples = int(round(float(separator_config["chunk_seconds"]) * RATE))

    results: dict[str, Any] = {}
    all_cases: list[dict[str, Any]] = []
    for mode in MODES:
        metrics, cases = evaluate_mode(
            mode=mode, questions=questions, components=components,
            scene_mixtures=scene_mixtures, separator=separator,
            label_to_id=label_to_id, chunk_samples=chunk_samples,
            device=device, batch_size=args.separator_batch_size,
        )
        results[mode] = metrics
        all_cases.extend(cases)
        print(json.dumps({mode: metrics}, ensure_ascii=False, sort_keys=True), flush=True)

    cases_path = output / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_cases),
        encoding="utf-8",
    )
    question_path = output / "question_predictions.jsonl"
    question_path.write_text(
        "".join(
            json.dumps(
                {
                    "item_id": q.item_id, "scene_id": q.scene_id, "question": q.question,
                    "gold_answer_label": q.answer_label_diagnostic_only,
                    "no_evidence": q.no_evidence, "predicted_none": q.predicted_none,
                    "predicted_label": q.predicted_label,
                    "predicted_label_confidence": q.predicted_label_confidence,
                    "oracle_span_predicted_label": q.oracle_span_predicted_label,
                    "oracle_span_predicted_label_confidence": q.oracle_span_predicted_label_confidence,
                    "gold_answer_span": q.gold_answer, "predicted_answer_span": q.predicted_span,
                    "anchor_iou": q.anchor_iou, "answer_iou": q.answer_iou,
                },
                ensure_ascii=False, sort_keys=True,
            ) + "\n"
            for q in questions
        ),
        encoding="utf-8",
    )
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "development integration; separator was checkpoint-selected on a 640-event subset of this dev domain",
        "warning": "Only predicted_label_predicted_span is deployable; oracle modes are diagnostic upper bounds.",
        "questions": len(questions),
        "scenes": len({q.scene_id for q in questions}),
        "classes": len(labels),
        "locked_objectness_threshold": threshold,
        "results": results,
        "artifacts": {
            "cases": {"path": str(cases_path), "sha256": _sha256_file(cases_path)},
            "question_predictions": {
                "path": str(question_path), "sha256": _sha256_file(question_path),
            },
        },
        "checkpoints": {
            "temporal": _sha256_file(args.temporal_checkpoint.resolve()),
            "semantic": _sha256_file(args.semantic_checkpoint.resolve()),
            "separator": _sha256_file(args.separator_checkpoint.resolve()),
            "label_clap": _sha256_file(args.label_embedding_cache.resolve()),
        },
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
