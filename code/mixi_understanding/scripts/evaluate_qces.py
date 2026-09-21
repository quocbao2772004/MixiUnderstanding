#!/usr/bin/env python3
"""Evaluate a QCES checkpoint and render question-swap listening examples."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from functools import partial
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixi_understanding.data.qces_schema import QCESRecord
from mixi_understanding.data.qces_v4_schema import QCESV4Record
from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
    NO_FOUNDATION_FEATURES,
)
from mixi_understanding.qces.beats_auditor import (
    BEATsAudioSetAuditor,
    score_qces_streams,
    summarize_beats_items,
)
from mixi_understanding.qces.data import QCESManifestDataset, collate_qces
from mixi_understanding.qces.metrics import (
    scale_dependent_sdr,
    scale_invariant_sdr,
)
from mixi_understanding.qces.model import load_qces_checkpoint
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    file_identity,
    source_tree_identity as audiosep_source_tree_identity,
)
from mixi_understanding.scripts.train_qces import (
    FoundationFeatureCache,
    add_foundation_features,
    load_foundation_feature_cache,
)


SUPPORTED_RECORD_TYPES = (QCESRecord, QCESV4Record, QCESV5Record)

# Keep the legacy ``summary`` object stable for existing v3 consumers while
# exposing an unambiguous, publication-table-ready view beside it.
SUMMARY_DIRECTIONS = {
    "evidence_l1": "↓",
    "residual_l1": "↓",
    "evidence_si_sdr_answerable": "↑",
    "evidence_si_sdr_median": "↑",
    "evidence_si_sdr_minimum": "↑",
    "evidence_si_sdri_answerable": "↑",
    "anchor_si_sdr_answerable": "↑",
    "answer_si_sdr_answerable": "↑",
    "weakest_role_si_sdr_answerable": "↑",
    "weakest_role_si_sdr_minimum": "↑",
    "evidence_sd_sdr_answerable": "↑",
    "evidence_sd_sdr_median": "↑",
    "evidence_sd_sdr_minimum": "↑",
    "evidence_sd_sdri_answerable": "↑",
    "anchor_sd_sdr_answerable": "↑",
    "answer_sd_sdr_answerable": "↑",
    "weakest_role_sd_sdr_answerable": "↑",
    "weakest_role_sd_sdr_minimum": "↑",
    "temporal_iou": "↑",
    "answerable_temporal_iou": "↑",
    "no_evidence_accuracy": "↑",
    "no_evidence_balanced_accuracy": "↑",
    "no_evidence_auroc": "↑",
    "no_evidence_f1": "↑",
    "no_evidence_recall": "↑",
    "answerable_false_silence_rate": "↓",
    "mean_answerable_question_contrast": "↑",
    "minimum_answerable_question_contrast": "↑",
    "mean_no_evidence_retained_ratio": "↓",
    "maximum_no_evidence_retained_ratio": "↓",
    "maximum_mixture_consistency_l1": "↓",
}


def summary_with_directions(summary: Mapping[str, float]) -> Dict[str, float]:
    """Suffix every reported scalar metric with its optimization direction."""

    missing = set(summary) - set(SUMMARY_DIRECTIONS)
    if missing:
        raise ValueError(f"metric directions are missing for: {sorted(missing)}")
    return {
        f"{name}_{SUMMARY_DIRECTIONS[name]}": value for name, value in summary.items()
    }


def binary_auroc(scores: Sequence[float], targets: Sequence[bool]) -> float:
    """Exact rank AUROC with average ranks for tied scores."""

    if len(scores) != len(targets) or not scores:
        raise ValueError("AUROC requires equally sized non-empty scores and targets")
    positives = sum(bool(target) for target in targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative examples")
    ordered = sorted(
        ((float(score), bool(target)) for score, target in zip(scores, targets)),
        key=lambda pair: pair[0],
    )
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        # Ranks are one-indexed. Every member of a tie receives the mean rank.
        average_rank = 0.5 * ((index + 1) + end)
        positive_rank_sum += average_rank * sum(
            target for _, target in ordered[index:end]
        )
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path)
    parser.add_argument("--audiosep-config", type=Path)
    parser.add_argument("--audiosep-checkpoint", type=Path)
    parser.add_argument(
        "--foundation-feature-cache",
        type=Path,
        help=(
            "Strict frozen AudioSep-CLAP cache for this exact evaluation "
            "manifest. Required only by audiosep_clap checkpoints."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--beats-root",
        type=Path,
        help="Pinned official Microsoft/unilm source root for evaluation-only BEATs.",
    )
    parser.add_argument(
        "--beats-checkpoint",
        type=Path,
        help="Pinned BEATs iter3+ AS2M fine-tuned cpt2 checkpoint.",
    )
    parser.add_argument(
        "--audioset-labels",
        type=Path,
        help="Pinned official AudioSet class_labels_indices.csv.",
    )
    parser.add_argument(
        "--beats-stream-batch-size",
        type=int,
        default=5,
        help="Streams per frozen BEATs forward; X,E*,R*,E,R total five.",
    )
    parser.add_argument(
        "--no-render-audio",
        action="store_true",
        help="Compute the full numeric report without writing per-item WAV files.",
    )
    parser.add_argument(
        "--render-item-id",
        action="append",
        default=[],
        help=(
            "Render WAVs only for this exact sample ID while still scoring the "
            "complete manifest. Repeat for a small leakage-safe preview. If "
            "omitted, every item is rendered unless --no-render-audio is set."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.no_render_audio and args.render_item_id:
        parser.error("--render-item-id cannot be combined with --no-render-audio")
    if len(args.render_item_id) != len(set(args.render_item_id)):
        parser.error("--render-item-id values must be unique")
    beats_paths = (args.beats_root, args.beats_checkpoint, args.audioset_labels)
    if any(path is not None for path in beats_paths) and not all(
        path is not None for path in beats_paths
    ):
        parser.error(
            "--beats-root, --beats-checkpoint, and --audioset-labels are all-or-none"
        )
    if args.beats_stream_batch_size <= 0:
        parser.error("--beats-stream-batch-size must be positive")
    return args


def checkpoint_foundation_feature_mode(
    checkpoint: Mapping[str, Any],
) -> str | None:
    """Read the opt-in mode without changing legacy checkpoint defaults."""

    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        return None
    mode = config.get("foundation_feature_mode", NO_FOUNDATION_FEATURES)
    return mode if isinstance(mode, str) else None


def validate_foundation_evaluation_args(
    args: argparse.Namespace, mode: str | None
) -> None:
    """Fail before model construction when frozen cache inputs are incomplete."""

    cache_dir = getattr(args, "foundation_feature_cache", None)
    if mode != AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        if cache_dir is not None:
            raise SystemExit(
                "--foundation-feature-cache is only valid for an "
                "audiosep_clap checkpoint"
            )
        return

    required = {
        "--foundation-feature-cache": cache_dir,
        "--audiosep-root": getattr(args, "audiosep_root", None),
        "--audiosep-checkpoint": getattr(args, "audiosep_checkpoint", None),
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise SystemExit("audiosep_clap evaluation requires " + ", ".join(missing))
    assert cache_dir is not None
    audiosep_root = args.audiosep_root
    audiosep_checkpoint = args.audiosep_checkpoint
    assert audiosep_root is not None and audiosep_checkpoint is not None
    if not cache_dir.resolve().is_dir():
        raise SystemExit(
            f"foundation-feature cache directory not found: {cache_dir.resolve()}"
        )
    if not audiosep_root.resolve().is_dir():
        raise SystemExit(f"AudioSep root not found: {audiosep_root.resolve()}")
    if not audiosep_checkpoint.resolve().is_file():
        raise SystemExit(
            f"AudioSep checkpoint not found: {audiosep_checkpoint.resolve()}"
        )


def load_evaluation_foundation_cache(
    args: argparse.Namespace,
    manifest: Path,
    records: Sequence[object],
) -> FoundationFeatureCache:
    """Load the strict manifest/asset-bound CLAP cache for evaluation."""

    cache_dir = args.foundation_feature_cache
    audiosep_root = args.audiosep_root
    audiosep_checkpoint = args.audiosep_checkpoint
    assert cache_dir is not None
    assert audiosep_root is not None
    assert audiosep_checkpoint is not None
    return load_foundation_feature_cache(
        cache_dir,
        manifest,
        records,
        "evaluation",
        audiosep_checkpoint_identity=file_identity(audiosep_checkpoint),
        audiosep_source_identity=audiosep_source_tree_identity(audiosep_root),
    )


def forward_evaluation_batch(
    model: Any,
    batch: Mapping[str, Any],
) -> Any:
    """Keep the historical call exact and require both CLAP tensors when opted in."""

    mode = getattr(
        getattr(model, "config", None),
        "foundation_feature_mode",
        NO_FOUNDATION_FEATURES,
    )
    if mode == NO_FOUNDATION_FEATURES:
        if "question_clap" in batch or "scene_clap" in batch:
            raise ValueError("legacy evaluation received foundation features")
        return model(batch["mixture"], batch["question_ids"], batch["question_mask"])
    if mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        question_clap = batch.get("question_clap")
        scene_clap = batch.get("scene_clap")
        if not isinstance(question_clap, torch.Tensor) or not isinstance(
            scene_clap, torch.Tensor
        ):
            raise ValueError(
                "audiosep_clap evaluation refuses a batch missing "
                "question_clap or scene_clap"
            )
        return model(
            batch["mixture"],
            batch["question_ids"],
            batch["question_mask"],
            question_clap=question_clap,
            scene_clap=scene_clap,
        )
    raise ValueError(f"unsupported foundation feature mode: {mode!r}")


def _role_windows_overlap(record: QCESV5Record) -> bool:
    return any(
        max(anchor[0], answer[0]) < min(anchor[1], answer[1])
        for anchor in record.anchor_intervals
        for answer in record.answer_intervals
    )


def v5_item_metadata(record: object) -> Dict[str, Any]:
    """Publication subgroup metadata, emitted only for QCES-v5 records."""

    if not isinstance(record, QCESV5Record):
        return {}
    same_role_label = None
    if not record.no_evidence:
        anchor_labels = tuple(
            record.event_by_id(event_id).label for event_id in record.anchor_event_ids
        )
        answer_labels = tuple(
            record.event_by_id(event_id).label for event_id in record.answer_event_ids
        )
        same_role_label = anchor_labels == answer_labels
    return {
        "scene_family_id": record.scene_family_id,
        "variant_id": record.variant_id,
        "counterfactual_group_id": record.counterfactual_group_id,
        "paraphrase_family_id": record.paraphrase_family_id,
        "question_semantics_id": record.question_semantics_id,
        "relation": record.relation,
        "evaluation_axis": record.evaluation_axis,
        "primary_counterfactual_probe": record.primary_counterfactual_probe,
        "same_role_label": same_role_label,
        "same_label_repeat": record.same_label_repeat,
        "role_windows_overlap": _role_windows_overlap(record),
        "semantic_overlap": record.semantic_overlap,
        "max_polyphony": record.max_polyphony,
        "hard_case_tags": list(record.hard_case_tags),
    }


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _normalized_distance(
    first: torch.Tensor, second: torch.Tensor, mixture: torch.Tensor
) -> float:
    scale = mixture.abs().mean().clamp_min(1e-8)
    return float(((first - second).abs().mean() / scale).cpu())


def _write_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    sf.write(
        path,
        waveform.detach().cpu().numpy(),
        sample_rate,
        subtype="PCM_16",
    )


def _active_intervals(
    active: torch.Tensor, hop_samples: int, sample_rate: int, duration: float
) -> List[List[float]]:
    flags = active.to(torch.bool).cpu().tolist()
    intervals: List[List[float]] = []
    start = None
    for index, enabled in enumerate(flags + [False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            intervals.append(
                [
                    round(start * hop_samples / sample_rate, 6),
                    round(min(index * hop_samples / sample_rate, duration), 6),
                ]
            )
            start = None
    return intervals


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or not 0.0 < args.threshold < 1.0:
        raise SystemExit("batch-size must be positive and threshold must be in (0, 1)")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    foundation_mode = (
        checkpoint_foundation_feature_mode(checkpoint)
        if isinstance(checkpoint, Mapping)
        else None
    )
    validate_foundation_evaluation_args(args, foundation_mode)
    model = load_qces_checkpoint(
        checkpoint,
        map_location=device,
        audiosep_repository_root=(
            str(args.audiosep_root.resolve()) if args.audiosep_root else None
        ),
        audiosep_config_path=(
            str(args.audiosep_config.resolve()) if args.audiosep_config else None
        ),
        audiosep_checkpoint_path=(
            str(args.audiosep_checkpoint.resolve())
            if args.audiosep_checkpoint
            else None
        ),
    ).eval()
    dataset = QCESManifestDataset(args.manifest.resolve(), crop_samples=None)
    if not all(
        isinstance(record, SUPPORTED_RECORD_TYPES) for record in dataset.records
    ):
        raise SystemExit(
            "question-swap evaluation requires a QCES v3, v4, or v5 manifest"
        )
    render_item_ids = set(args.render_item_id)
    available_item_ids = {record.sample_id for record in dataset.records}
    unknown_render_ids = sorted(render_item_ids - available_item_ids)
    if unknown_render_ids:
        raise SystemExit(
            "render item IDs are absent from the evaluation manifest: "
            + ", ".join(unknown_render_ids)
        )
    foundation_cache = None
    if foundation_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        foundation_cache = load_evaluation_foundation_cache(
            args,
            args.manifest.resolve(),
            dataset.records,
        )
    if foundation_mode != model.config.foundation_feature_mode:
        raise RuntimeError(
            "checkpoint foundation feature mode changed during model loading"
        )
    is_v5_manifest = all(isinstance(record, QCESV5Record) for record in dataset.records)
    beats_auditor = None
    if args.beats_root is not None:
        if not is_v5_manifest:
            raise SystemExit("the BEATs evidence auditor requires a QCES-v5 manifest")
        assert args.beats_checkpoint is not None and args.audioset_labels is not None
        qces_labels = {
            event.label
            for record in dataset.records
            if isinstance(record, QCESV5Record)
            for event in record.events
        }
        beats_auditor = BEATsAudioSetAuditor.from_assets(
            args.beats_root,
            args.beats_checkpoint,
            args.audioset_labels,
            qces_labels,
            device,
        )
    tokenizer = StableHashTokenizer(
        model.config.vocab_size, model.config.max_question_tokens
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_qces, tokenizer=tokenizer),
    )

    rendered: List[Dict[str, Any]] = []
    global_index = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            add_foundation_features(batch, foundation_cache, device)
            output = forward_evaluation_batch(model, batch)
            frames = output.composition.evidence_probability.size(-1)
            anchor = F.adaptive_max_pool1d(
                batch["anchor_mask"][:, None], frames
            ).squeeze(1)
            answer = F.adaptive_max_pool1d(
                batch["answer_mask"][:, None], frames
            ).squeeze(1)
            role_target = (anchor + answer) > 0.5
            role_prediction = output.composition.evidence_probability >= args.threshold
            for local_index in range(batch["mixture"].size(0)):
                record = dataset.records[global_index]
                assert isinstance(record, SUPPORTED_RECORD_TYPES)
                mixture = batch["mixture"][local_index]
                evidence = output.evidence[local_index]
                residual = output.residual[local_index]
                evidence_target = batch["evidence"][local_index]
                residual_target = batch["residual"][local_index]
                intersection = (
                    role_prediction[local_index] & role_target[local_index]
                ).sum()
                union = (role_prediction[local_index] | role_target[local_index]).sum()
                temporal_iou = float(intersection / union) if int(union) > 0 else 1.0
                no_evidence_probability = float(
                    output.composition.no_evidence_logit[local_index].sigmoid()
                )
                item: Dict[str, Any] = {
                    "id": record.sample_id,
                    "scene_id": record.scene_id,
                    "question_index": record.question_index,
                    "question_type": record.question_type,
                    "question": record.question,
                    "answer": record.answer,
                    "no_evidence": record.no_evidence,
                    "no_evidence_probability": no_evidence_probability,
                    "predicted_evidence_intervals": _active_intervals(
                        role_prediction[local_index],
                        output.composition.frame_hop_samples,
                        record.sample_rate,
                        record.duration_seconds,
                    ),
                    "target_evidence_intervals": [
                        list(interval)
                        for interval in record.anchor_intervals
                        + record.answer_intervals
                    ],
                    "no_evidence_correct": (no_evidence_probability >= args.threshold)
                    == record.no_evidence,
                    "evidence_l1": float((evidence - evidence_target).abs().mean()),
                    "residual_l1": float((residual - residual_target).abs().mean()),
                    "temporal_iou": temporal_iou,
                    "retained_ratio": float(
                        evidence.abs().sum() / mixture.abs().sum().clamp_min(1e-8)
                    ),
                    "target_retained_ratio": float(
                        evidence_target.abs().sum()
                        / mixture.abs().sum().clamp_min(1e-8)
                    ),
                    "mixture_consistency_l1": float(
                        (evidence + residual - mixture).abs().mean()
                    ),
                }
                semantic_candidate_weight = getattr(
                    output.composition,
                    "foundation_semantic_candidate_weight",
                    None,
                )
                item["foundation_semantic_candidate_weight_descriptive"] = (
                    float(semantic_candidate_weight)
                    if semantic_candidate_weight is not None
                    else None
                )
                item.update(v5_item_metadata(record))
                if beats_auditor is not None and not record.no_evidence:
                    assert isinstance(record, QCESV5Record)
                    probabilities = beats_auditor.score(
                        torch.stack(
                            (
                                mixture,
                                evidence_target,
                                residual_target,
                                evidence,
                                residual,
                            )
                        ),
                        record.sample_rate,
                        args.beats_stream_batch_size,
                    )
                    required_labels = [
                        record.event_by_id(event_id).label
                        for event_id in record.evidence_event_ids
                    ]
                    required_set = set(required_labels)
                    evidence_event_ids = set(record.evidence_event_ids)
                    residual_event_labels = sorted(
                        {
                            event.label
                            for event in record.events
                            if event.event_id not in evidence_event_ids
                        }
                    )
                    excluded_labels = sorted(
                        {event.label for event in record.events} - required_set
                    )
                    item["beats_audioset"] = score_qces_streams(
                        probabilities,
                        required_labels,
                        excluded_labels,
                        beats_auditor.label_indices,
                        residual_event_labels,
                    )
                elif beats_auditor is not None:
                    item["beats_audioset"] = None
                if not record.no_evidence:
                    item["evidence_si_sdr"] = float(
                        scale_invariant_sdr(evidence[None], evidence_target[None])[0]
                    )
                    item["mixture_si_sdr"] = float(
                        scale_invariant_sdr(mixture[None], evidence_target[None])[0]
                    )
                    item["evidence_si_sdri"] = (
                        item["evidence_si_sdr"] - item["mixture_si_sdr"]
                    )
                    predicted_anchor = evidence * batch["anchor_mask"][local_index]
                    predicted_answer = evidence * batch["answer_mask"][local_index]
                    target_anchor = batch["anchor_stem"][local_index]
                    target_answer = batch["answer_stem"][local_index]
                    item["anchor_si_sdr"] = float(
                        scale_invariant_sdr(
                            predicted_anchor[None], target_anchor[None]
                        )[0]
                    )
                    item["answer_si_sdr"] = float(
                        scale_invariant_sdr(
                            predicted_answer[None], target_answer[None]
                        )[0]
                    )
                    item["weakest_role_si_sdr"] = min(
                        item["anchor_si_sdr"], item["answer_si_sdr"]
                    )
                    if is_v5_manifest:
                        item["evidence_sd_sdr"] = float(
                            scale_dependent_sdr(evidence[None], evidence_target[None])[
                                0
                            ]
                        )
                        item["mixture_sd_sdr"] = float(
                            scale_dependent_sdr(mixture[None], evidence_target[None])[0]
                        )
                        item["evidence_sd_sdri"] = (
                            item["evidence_sd_sdr"] - item["mixture_sd_sdr"]
                        )
                        item["anchor_sd_sdr"] = float(
                            scale_dependent_sdr(
                                predicted_anchor[None], target_anchor[None]
                            )[0]
                        )
                        item["answer_sd_sdr"] = float(
                            scale_dependent_sdr(
                                predicted_answer[None], target_answer[None]
                            )[0]
                        )
                        item["weakest_role_sd_sdr"] = min(
                            item["anchor_sd_sdr"], item["answer_sd_sdr"]
                        )
                else:
                    item["evidence_si_sdr"] = None
                    item["mixture_si_sdr"] = None
                    item["evidence_si_sdri"] = None
                    item["anchor_si_sdr"] = None
                    item["answer_si_sdr"] = None
                    item["weakest_role_si_sdr"] = None
                    if is_v5_manifest:
                        item["evidence_sd_sdr"] = None
                        item["mixture_sd_sdr"] = None
                        item["evidence_sd_sdri"] = None
                        item["anchor_sd_sdr"] = None
                        item["answer_sd_sdr"] = None
                        item["weakest_role_sd_sdr"] = None

                if not args.no_render_audio and (
                    not render_item_ids or record.sample_id in render_item_ids
                ):
                    scene_dir = output_dir / record.scene_id
                    question_dir = scene_dir / (
                        f"q{record.question_index}_{record.question_type}"
                    )
                    question_dir.mkdir(parents=True, exist_ok=True)
                    mixture_path = scene_dir / "mixture.wav"
                    if not mixture_path.exists():
                        _write_audio(mixture_path, mixture, record.sample_rate)
                    _write_audio(
                        question_dir / "predicted_evidence.wav",
                        evidence,
                        record.sample_rate,
                    )
                    _write_audio(
                        question_dir / "target_evidence.wav",
                        evidence_target,
                        record.sample_rate,
                    )
                    _write_audio(
                        question_dir / "predicted_residual.wav",
                        residual,
                        record.sample_rate,
                    )
                    _write_audio(
                        question_dir / "target_residual.wav",
                        residual_target,
                        record.sample_rate,
                    )
                    if not record.no_evidence:
                        _write_audio(
                            question_dir / "predicted_anchor_role.wav",
                            predicted_anchor,
                            record.sample_rate,
                        )
                        _write_audio(
                            question_dir / "target_anchor_role.wav",
                            target_anchor,
                            record.sample_rate,
                        )
                        _write_audio(
                            question_dir / "predicted_answer_role.wav",
                            predicted_answer,
                            record.sample_rate,
                        )
                        _write_audio(
                            question_dir / "target_answer_role.wav",
                            target_answer,
                            record.sample_rate,
                        )
                    (question_dir / "metadata.json").write_text(
                        json.dumps(item, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                item["predicted_evidence_tensor"] = evidence.cpu()
                item["target_evidence_tensor"] = evidence_target.cpu()
                item["mixture_tensor"] = mixture.cpu()
                rendered.append(item)
                global_index += 1

    scenes: List[Dict[str, Any]] = []
    for scene_id in sorted({item["scene_id"] for item in rendered}):
        items = [item for item in rendered if item["scene_id"] == scene_id]
        answerable = [item for item in items if not item["no_evidence"]]
        predicted_distances = [
            _normalized_distance(
                first["predicted_evidence_tensor"],
                second["predicted_evidence_tensor"],
                first["mixture_tensor"],
            )
            for first, second in combinations(answerable, 2)
        ]
        target_distances = [
            _normalized_distance(
                first["target_evidence_tensor"],
                second["target_evidence_tensor"],
                first["mixture_tensor"],
            )
            for first, second in combinations(answerable, 2)
        ]
        negative = [item for item in items if item["no_evidence"]]
        if not negative:
            raise RuntimeError(f"scene {scene_id} has no no-evidence question")
        negative_retained_ratios = [item["retained_ratio"] for item in negative]
        scenes.append(
            {
                "scene_id": scene_id,
                "answerable_pairwise_predicted_contrast": predicted_distances,
                "answerable_pairwise_target_contrast": target_distances,
                "minimum_predicted_contrast": min(predicted_distances),
                "mean_predicted_contrast": _mean(predicted_distances),
                "mean_target_contrast": _mean(target_distances),
                # This remains numerically identical for v3 (one negative per
                # scene) and correctly aggregates all v4 negatives.
                "no_evidence_retained_ratio": _mean(negative_retained_ratios),
                "maximum_no_evidence_retained_ratio": max(negative_retained_ratios),
                "no_evidence_questions": len(negative),
            }
        )

    serializable = []
    for item in rendered:
        serializable.append(
            {key: value for key, value in item.items() if not key.endswith("_tensor")}
        )
    answerable_si_sdr = [
        item["evidence_si_sdr"]
        for item in serializable
        if item["evidence_si_sdr"] is not None
    ]
    answerable_si_sdri = [
        item["evidence_si_sdri"]
        for item in serializable
        if item["evidence_si_sdri"] is not None
    ]
    anchor_si_sdr = [
        item["anchor_si_sdr"]
        for item in serializable
        if item["anchor_si_sdr"] is not None
    ]
    answer_si_sdr = [
        item["answer_si_sdr"]
        for item in serializable
        if item["answer_si_sdr"] is not None
    ]
    weakest_role_si_sdr = [
        item["weakest_role_si_sdr"]
        for item in serializable
        if item["weakest_role_si_sdr"] is not None
    ]
    answerable_sd_sdr = (
        [
            item["evidence_sd_sdr"]
            for item in serializable
            if item["evidence_sd_sdr"] is not None
        ]
        if is_v5_manifest
        else []
    )
    answerable_sd_sdri = (
        [
            item["evidence_sd_sdri"]
            for item in serializable
            if item["evidence_sd_sdri"] is not None
        ]
        if is_v5_manifest
        else []
    )
    anchor_sd_sdr = (
        [
            item["anchor_sd_sdr"]
            for item in serializable
            if item["anchor_sd_sdr"] is not None
        ]
        if is_v5_manifest
        else []
    )
    answer_sd_sdr = (
        [
            item["answer_sd_sdr"]
            for item in serializable
            if item["answer_sd_sdr"] is not None
        ]
        if is_v5_manifest
        else []
    )
    weakest_role_sd_sdr = (
        [
            item["weakest_role_sd_sdr"]
            for item in serializable
            if item["weakest_role_sd_sdr"] is not None
        ]
        if is_v5_manifest
        else []
    )
    no_evidence_targets = [bool(item["no_evidence"]) for item in serializable]
    no_evidence_scores = [
        float(item["no_evidence_probability"]) for item in serializable
    ]
    no_evidence_predictions = [score >= args.threshold for score in no_evidence_scores]
    no_evidence_positive_count = sum(no_evidence_targets)
    answerable_count = len(no_evidence_targets) - no_evidence_positive_count
    if no_evidence_positive_count == 0 or answerable_count == 0:
        raise RuntimeError(
            "evaluation requires both answerable and no-evidence records"
        )
    true_no_evidence = sum(
        prediction and target
        for prediction, target in zip(no_evidence_predictions, no_evidence_targets)
    )
    false_no_evidence = sum(
        prediction and not target
        for prediction, target in zip(no_evidence_predictions, no_evidence_targets)
    )
    correctly_answerable = sum(
        not prediction and not target
        for prediction, target in zip(no_evidence_predictions, no_evidence_targets)
    )
    no_evidence_recall = true_no_evidence / no_evidence_positive_count
    answerable_recall = correctly_answerable / answerable_count
    no_evidence_precision = true_no_evidence / max(
        true_no_evidence + false_no_evidence, 1
    )
    no_evidence_f1 = (
        2.0
        * no_evidence_precision
        * no_evidence_recall
        / max(no_evidence_precision + no_evidence_recall, 1e-12)
    )
    summary = {
        "evidence_l1": _mean([item["evidence_l1"] for item in serializable]),
        "residual_l1": _mean([item["residual_l1"] for item in serializable]),
        "evidence_si_sdr_answerable": _mean(answerable_si_sdr),
        "evidence_si_sdr_median": (
            float(median(answerable_si_sdr)) if answerable_si_sdr else 0.0
        ),
        "evidence_si_sdr_minimum": (
            min(answerable_si_sdr) if answerable_si_sdr else 0.0
        ),
        "evidence_si_sdri_answerable": _mean(answerable_si_sdri),
        "anchor_si_sdr_answerable": _mean(anchor_si_sdr),
        "answer_si_sdr_answerable": _mean(answer_si_sdr),
        "weakest_role_si_sdr_answerable": _mean(weakest_role_si_sdr),
        "weakest_role_si_sdr_minimum": (
            min(weakest_role_si_sdr) if weakest_role_si_sdr else 0.0
        ),
        # Retained for backward compatibility; the answerable-only value is
        # the scientifically cleaner result because an empty target otherwise
        # gives a no-evidence example IoU=1 when prediction is also empty.
        "temporal_iou": _mean([item["temporal_iou"] for item in serializable]),
        "answerable_temporal_iou": _mean(
            [item["temporal_iou"] for item in serializable if not item["no_evidence"]]
        ),
        "no_evidence_accuracy": _mean(
            [float(item["no_evidence_correct"]) for item in serializable]
        ),
        "no_evidence_balanced_accuracy": 0.5 * (no_evidence_recall + answerable_recall),
        "no_evidence_auroc": binary_auroc(no_evidence_scores, no_evidence_targets),
        "no_evidence_f1": no_evidence_f1,
        "no_evidence_recall": no_evidence_recall,
        "answerable_false_silence_rate": 1.0 - answerable_recall,
        "mean_answerable_question_contrast": _mean(
            [scene["mean_predicted_contrast"] for scene in scenes]
        ),
        "minimum_answerable_question_contrast": min(
            scene["minimum_predicted_contrast"] for scene in scenes
        ),
        "mean_no_evidence_retained_ratio": _mean(
            [scene["no_evidence_retained_ratio"] for scene in scenes]
        ),
        "maximum_no_evidence_retained_ratio": max(
            scene["maximum_no_evidence_retained_ratio"] for scene in scenes
        ),
        "maximum_mixture_consistency_l1": max(
            item["mixture_consistency_l1"] for item in serializable
        ),
    }
    if is_v5_manifest:
        summary.update(
            {
                "evidence_sd_sdr_answerable": _mean(answerable_sd_sdr),
                "evidence_sd_sdr_median": (
                    float(median(answerable_sd_sdr)) if answerable_sd_sdr else 0.0
                ),
                "evidence_sd_sdr_minimum": (
                    min(answerable_sd_sdr) if answerable_sd_sdr else 0.0
                ),
                "evidence_sd_sdri_answerable": _mean(answerable_sd_sdri),
                "anchor_sd_sdr_answerable": _mean(anchor_sd_sdr),
                "answer_sd_sdr_answerable": _mean(answer_sd_sdr),
                "weakest_role_sd_sdr_answerable": _mean(weakest_role_sd_sdr),
                "weakest_role_sd_sdr_minimum": (
                    min(weakest_role_sd_sdr) if weakest_role_sd_sdr else 0.0
                ),
            }
        )
    report = {
        "format": "qces_question_swap_eval_v1",
        "schema_version": dataset.records[0].schema_version,
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": (
            hashlib.sha256(args.manifest.resolve().read_bytes()).hexdigest()
            if args.manifest.resolve().is_file()
            else None
        ),
        "device": str(device),
        "records": len(serializable),
        "counts": {
            "answerable": sum(not item["no_evidence"] for item in serializable),
            "no_evidence": sum(item["no_evidence"] for item in serializable),
        },
        "audio_rendering": {
            "enabled": not args.no_render_audio,
            "selection": "explicit_item_ids" if render_item_ids else "all_items",
            "requested_item_ids": sorted(render_item_ids),
            "rendered_items": (
                len(render_item_ids)
                if render_item_ids
                else (0 if args.no_render_audio else len(serializable))
            ),
        },
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "summary": summary,
        "summary_with_directions": summary_with_directions(summary),
        "controller_diagnostics": {
            "foundation_semantic_mixing_mode": (
                getattr(
                    model.config,
                    "foundation_semantic_mixing_mode",
                    LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
                )
            ),
            "foundation_semantic_candidate_weight_descriptive": (
                serializable[0]["foundation_semantic_candidate_weight_descriptive"]
                if serializable
                else None
            ),
            "interpretation": (
                "larger means more reliance on the learned semantic candidate; "
                "descriptive, not intrinsically better or worse"
            ),
        },
        "scenes": scenes,
        "items": serializable,
    }
    if beats_auditor is not None:
        beats_items = [
            item["beats_audioset"]
            for item in serializable
            if isinstance(item.get("beats_audioset"), Mapping)
        ]
        report["external_auditors"] = {
            "beats_audioset": {
                **summarize_beats_items(beats_items),
                "provenance": beats_auditor.provenance,
                "stream_order": [
                    "mixture",
                    "oracle_evidence",
                    "oracle_residual",
                    "predicted_evidence",
                    "predicted_residual",
                ],
                "evaluation_only": True,
                "model_parameters_updated_↓": 0,
                "qces_inputs_changed_↓": 0,
            }
        }
    if foundation_cache is not None:
        report["foundation_features"] = {
            "mode": AUDIOSEP_CLAP_FOUNDATION_FEATURES,
            "cache_identity": dict(foundation_cache.identity),
        }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary_with_directions"], indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
