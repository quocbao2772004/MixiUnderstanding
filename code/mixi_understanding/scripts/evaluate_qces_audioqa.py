#!/usr/bin/env python3
"""Audit QCES acoustic evidence with a frozen audio-language QA model.

This is deliberately an evaluation-only program.  It never trains or updates
the separator or the QA auditor.  For every QCES v4/v5 question it scores the five
answer-option letters under a set of paired acoustic interventions:

* original mixture (X), predicted evidence (E), predicted residual (R),
* oracle evidence (E*) and oracle residual (R*),
* silence, shuffled predicted evidence, cross-family shuffled oracle evidence,
  and a question-only control.

The primary path uses conditional option log probabilities rather than free
generation.  Results are appended one record/condition at a time to a JSONL
file, so an interrupted 8B-model run resumes without repeating finished work.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from mixi_understanding.data.qces_v4_schema import QCESV4Record, parse_qces_v4_record
from mixi_understanding.data.qces_v5_schema import (
    DERIVED_STORAGE_MODE,
    SCHEMA_VERSIONS as QCES_V5_SCHEMA_VERSIONS,
    QCESV5Record,
    parse_qces_v5_record,
)
from mixi_understanding.qces.data import read_jsonl


FORMAT_VERSION = "qces_audioqa_audit_v7"
PROMPT_VERSION = "qces_mc_letter_v1"
SCORING_VERSION = "qces_exact_letter_conditional_logprob_v2"
AUDITOR_CHOICES = ("af3", "qwen2_audio", "phi4mm")
OPTION_LABELS = ("A", "B", "C", "D", "E")
ALL_CONDITIONS = (
    "mixture",
    "predicted_evidence",
    "predicted_residual",
    "oracle_evidence",
    "oracle_residual",
    "silence",
    "shuffled_evidence",
    "shuffled_oracle_evidence",
    "question_only",
)
OPTION_ORDER_CONTROL_SUFFIX = "__option_order_permuted"
PREDICTED_CONDITIONS = {
    "predicted_evidence",
    "predicted_residual",
    "shuffled_evidence",
}
SUPPORTED_RECORD_TYPES = (QCESV4Record, QCESV5Record)
AudioQARecord = QCESV4Record | QCESV5Record


CONDITION_METRIC_DEFINITIONS = {
    "multiple_choice_accuracy_all_↑": "↑ Higher is better: exact five-option accuracy on all records.",
    "answerable_accuracy_↑": "↑ Higher is better: exact accuracy on answerable records only.",
    "no_evidence_accuracy_↑": "↑ Higher is better: correct no_evidence decisions on absent-anchor records.",
    "answerability_balanced_accuracy_↑": "↑ Higher is better: mean no-evidence recall and answerable recall.",
    "no_evidence_false_positive_rate_on_answerable_↓": "↓ Lower is better: answerable records incorrectly predicted as no_evidence.",
    "false_answer_rate_on_no_evidence_↓": "↓ Lower is better: absent-anchor records assigned a non-no_evidence answer.",
    "mean_gold_option_probability_↑": "↑ Higher is better: mean softmax probability of the gold option.",
    "candidate_aware_chance_accuracy_↑": "↑ Reference floor, not a system target: 0.5 for first and 0.2 otherwise.",
    "answerable_candidate_aware_chance_accuracy_↑": "↑ Reference floor on answerable records only: 0.5 for first and 0.2 otherwise.",
    "accuracy_by_relation_↑": "↑ Higher is better: exact accuracy separately for after, before, and first.",
    "answerable_accuracy_by_relation_↑": "↑ Higher is better: exact answerable-only accuracy separately for after, before, and first.",
    "no_evidence_accuracy_by_relation_↑": "↑ Higher is better: exact no-evidence-only accuracy separately for after, before, and first.",
    "first_question_named_candidate_prediction_rate_↑": "↑ Higher is better: predictions on first questions select one of the two candidates named in the question.",
    "first_question_first_mention_bias_gap_↓": "↓ Lower is better: absolute gap between predicted and gold first-mentioned-candidate rates on first questions.",
    "first_question_mention_position_accuracy_gap_↓": "↓ Lower is better: absolute accuracy gap between first questions whose gold answer is mentioned first versus second.",
    "first_question_surface_pair_prediction_invariance_↑": "↑ Higher is better: paired first-question controls predict the same semantic answer after reversing candidate mentions and independently reshuffling option positions.",
    "first_question_surface_pair_both_correct_rate_↑": "↑ Higher is better: fraction of complete first-question surface-control pairs answered correctly in both renderings.",
    "first_question_surface_pair_at_least_one_correct_rate_↑": "↑ Higher is better: fraction of complete first-question surface-control pairs answered correctly in at least one rendering.",
    "first_question_surface_pair_gold_log_score_absolute_gap_↓": "↓ Lower is better: mean absolute paired gap in the gold option log score after the combined mention-order and option-order intervention.",
    "first_question_surface_pair_gold_probability_absolute_gap_↓": "↓ Lower is better: mean absolute paired gap in the gold option probability after the combined mention-order and option-order intervention.",
}

PAIRED_METRIC_DEFINITIONS = {
    "predicted_evidence_sufficiency_accuracy_↑": "↑ Higher is better: QA accuracy on predicted evidence for answerable records.",
    "conditional_sufficiency_given_mixture_correct_↑": "↑ Higher is better: predicted evidence stays correct when the mixture was correct.",
    "predicted_evidence_accuracy_gain_over_mixture_↑": "↑ Higher is better: paired E accuracy minus X accuracy.",
    "predicted_evidence_gold_log_score_gain_over_mixture_↑": "↑ Higher is better: paired gold-option log-score gain E minus X.",
    "predicted_residual_answer_leakage_accuracy_↓": "↓ Lower is better: answer remains recoverable from predicted residual.",
    "conditional_residual_leakage_given_mixture_correct_↓": "↓ Lower is better: residual remains correct among mixture-correct questions.",
    "predicted_necessity_success_given_mixture_correct_↑": "↑ Higher is better: residual becomes wrong among mixture-correct questions.",
    "predicted_necessity_accuracy_drop_↑": "↑ Higher is better: paired X accuracy minus R accuracy.",
    "predicted_necessity_gold_log_score_drop_↑": "↑ Higher is better: paired gold-option log-score drop X minus R.",
    "oracle_evidence_sufficiency_accuracy_↑": "↑ Higher is better: QA accuracy on oracle evidence for answerable records.",
    "mixture_accuracy_gain_over_question_only_↑": "↑ Higher is better for auditor calibration: paired mixture accuracy minus question-only accuracy.",
    "mixture_gold_log_score_gain_over_question_only_↑": "↑ Higher is better for auditor calibration: paired gold-option log-score gain for mixture over question-only.",
    "oracle_evidence_accuracy_gain_over_question_only_↑": "↑ Higher is better for auditor calibration: paired oracle-E accuracy minus question-only accuracy.",
    "oracle_evidence_gold_log_score_gain_over_question_only_↑": "↑ Higher is better for auditor calibration: paired gold-option log-score gain for oracle E over question-only.",
    "oracle_evidence_accuracy_gain_over_shuffled_oracle_evidence_↑": "↑ Higher is better for auditor calibration: paired oracle-E accuracy minus cross-family shuffled-oracle-E accuracy.",
    "oracle_evidence_gold_log_score_gain_over_shuffled_oracle_evidence_↑": "↑ Higher is better for auditor calibration: paired gold-option log-score gain for oracle E over cross-family shuffled oracle E.",
    "oracle_evidence_accuracy_gain_over_oracle_residual_↑": "↑ Higher is better for auditor calibration: paired oracle-E accuracy minus oracle-R accuracy.",
    "oracle_evidence_gold_log_score_gain_over_oracle_residual_↑": "↑ Higher is better for auditor calibration: paired gold-option log-score gain for oracle E over oracle R.",
    "oracle_residual_answer_leakage_accuracy_↓": "↓ Lower is better: answer remains recoverable from oracle residual.",
    "oracle_necessity_success_given_mixture_correct_↑": "↑ Higher is better: oracle residual becomes wrong among mixture-correct questions.",
    "absolute_oracle_predicted_evidence_accuracy_gap_↓": "↓ Lower is better: absolute accuracy gap between oracle and predicted evidence.",
    "positive_oracle_evidence_headroom_accuracy_↓": "↓ Lower is better: positive part of oracle E* accuracy minus predicted E accuracy.",
    "predicted_evidence_no_evidence_accuracy_↑": "↑ Higher is better: predicted E selects no_evidence on absent-anchor questions.",
    "predicted_residual_no_evidence_accuracy_↑": "↑ Higher is better: predicted R selects no_evidence on absent-anchor questions.",
    "oracle_evidence_no_evidence_accuracy_↑": "↑ Higher is better: oracle E* selects no_evidence on absent-anchor questions.",
    "oracle_residual_no_evidence_accuracy_↑": "↑ Higher is better: oracle R* selects no_evidence on absent-anchor questions.",
    "silence_control_answerable_accuracy_↓": "↓ Lower is better for this shortcut control: answerable accuracy with silent audio.",
    "question_only_control_answerable_accuracy_↓": "↓ Lower is better for this shortcut control: answerable accuracy without audio.",
    "shuffled_evidence_control_answerable_accuracy_↓": "↓ Lower is better for this causal control: accuracy with another scene's E.",
    "shuffled_oracle_evidence_control_answerable_accuracy_↓": "↓ Lower is better for auditor calibration: answerable accuracy with oracle evidence from another independent scene family.",
    "predicted_evidence_accuracy_gain_over_shuffled_evidence_↑": "↑ Higher is better: paired predicted-E accuracy minus shuffled-E accuracy.",
    "predicted_evidence_gold_log_score_gain_over_shuffled_evidence_↑": "↑ Higher is better: paired gold-option log-score gain for predicted E over shuffled E.",
    "conditional_sufficiency_given_mixture_correct_question_only_wrong_↑": "↑ Higher is better: predicted evidence stays correct on audio-dependent records where mixture is correct and question-only is wrong.",
    "conditional_residual_leakage_given_mixture_correct_question_only_wrong_↓": "↓ Lower is better: predicted residual stays correct on audio-dependent records where mixture is correct and question-only is wrong.",
    "conditional_necessity_success_given_mixture_correct_question_only_wrong_↑": "↑ Higher is better: predicted residual becomes wrong on audio-dependent records where mixture is correct and question-only is wrong.",
    "oracle_conditional_sufficiency_given_mixture_correct_question_only_wrong_↑": "↑ Higher is better: oracle evidence stays correct on audio-dependent records where mixture is correct and question-only is wrong.",
    "oracle_conditional_residual_leakage_given_mixture_correct_question_only_wrong_↓": "↓ Lower is better: oracle residual stays correct on audio-dependent records where mixture is correct and question-only is wrong.",
    "oracle_conditional_necessity_success_given_mixture_correct_question_only_wrong_↑": "↑ Higher is better: oracle residual becomes wrong on audio-dependent records where mixture is correct and question-only is wrong.",
}

OPTION_ORDER_METRIC_DEFINITIONS = {
    "option_order_gold_position_changed_rate_↑": "↑ Higher is better for control integrity: fraction of pairs in which the gold answer occupies a different option position; the registered control requires 1.0.",
    "option_order_semantic_prediction_invariance_↑": "↑ Higher is better: the semantic prediction is unchanged when only answer-option positions are permuted.",
    "option_order_both_correct_rate_↑": "↑ Higher is better: both the registered and option-permuted presentations are answered correctly.",
    "option_order_at_least_one_correct_rate_↑": "↑ Higher is better: at least one of the two option-order presentations is answered correctly.",
    "option_order_accuracy_absolute_gap_↓": "↓ Lower is better: absolute accuracy difference between registered and option-permuted presentations.",
    "option_order_gold_log_score_absolute_gap_↓": "↓ Lower is better: mean paired absolute change in the gold-option log score under option-only permutation.",
    "option_order_gold_probability_absolute_gap_↓": "↓ Lower is better: mean paired absolute change in the gold-option probability under option-only permutation.",
}


@dataclass(frozen=True)
class InputDescriptor:
    """Immutable description of one acoustic intervention."""

    record_id: str
    condition: str
    source_record_id: str
    path: Optional[str]
    sha256: str
    sample_rate: int
    num_samples: int
    special: Optional[str] = None
    component_paths: Tuple[str, ...] = ()
    component_sha256s: Tuple[str, ...] = ()
    derivation_recipe: Optional[str] = None


@dataclass(frozen=True)
class OptionScore:
    """Five-way option scores returned by a frozen QA auditor."""

    log_scores: Tuple[float, ...]
    probabilities: Tuple[float, ...]
    predicted_index: int
    scoring_method: str
    candidate_token_lengths: Tuple[int, ...]


class OptionScorer(Protocol):
    def score(
        self,
        question: str,
        options: Sequence[str],
        waveform: Optional[np.ndarray],
        sample_rate: Optional[int],
    ) -> OptionScore:
        """Return one score per option; ``waveform=None`` means question-only."""

    def provenance(self) -> Mapping[str, Any]:
        """Return resolved model/runtime provenance after model loading."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Root used to resolve manifest-relative audio paths. Defaults to the "
            "manifest directory; set this for fingerprinted subset manifests."
        ),
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        help=(
            "Root produced by evaluate_qces.py. Required for predicted E/R and "
            "shuffled-evidence conditions."
        ),
    )
    parser.add_argument(
        "--predictions-report",
        type=Path,
        help=(
            "Optional explicit evaluation_report.json when predicted WAVs live "
            "inside one mode subdirectory (for SAM, temporal, or planner baselines)."
        ),
    )
    parser.add_argument(
        "--predictions-mode",
        help=(
            "Mode name to select from a multi-mode baseline report. The "
            "predictions root must point directly to that mode's scene folders."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Local HF directory or official model repository ID.",
    )
    parser.add_argument(
        "--auditor",
        choices=AUDITOR_CHOICES,
        default="af3",
        help=(
            "Frozen audio-language auditor implementation. Use qwen2_audio as an "
            "additional held-out checkpoint or phi4mm for a cross-architecture "
            "Phi/Conformer audit."
        ),
    )
    parser.add_argument(
        "--revision",
        help="Pinned HF commit/tag. Required for a remote repository ID.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ALL_CONDITIONS,
        default=list(ALL_CONDITIONS),
    )
    parser.add_argument(
        "--option-order-control-conditions",
        nargs="*",
        choices=ALL_CONDITIONS,
        default=(),
        help=(
            "Selected audio conditions to score a second time with identical "
            "audio/question and a deterministic gold-independent cyclic option "
            "permutation. Disabled by default because it adds one model call per "
            "record and selected condition."
        ),
    )
    parser.add_argument(
        "--split",
        choices=(
            "all",
            "train",
            "val",
            "test",
            "test_iid",
            "test_compositional_ood",
            "test_label_ood",
        ),
        default="all",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--quantization", choices=("none", "4bit", "8bit"), default="4bit"
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--device-map", default="auto", help="Accelerate device map for k-bit loading."
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--hash-model-weights",
        action="store_true",
        help="SHA256 every local safetensors/bin shard (slow, strongest local provenance).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate/hash every input and print the run fingerprint without "
            "loading the selected auditor."
        ),
    )
    args = parser.parse_args(argv)
    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    audio_conditions = tuple(dict.fromkeys(args.conditions))
    option_controls = tuple(dict.fromkeys(args.option_order_control_conditions))
    missing_control_bases = set(option_controls) - set(audio_conditions)
    if missing_control_bases:
        parser.error(
            "option-order controls require their base --conditions: "
            + ", ".join(sorted(missing_control_bases))
        )
    args.audio_conditions = audio_conditions
    args.option_order_control_conditions = option_controls
    args.conditions = audio_conditions + tuple(
        condition + OPTION_ORDER_CONTROL_SUFFIX for condition in option_controls
    )
    selected_audio_conditions = {
        condition.removesuffix(OPTION_ORDER_CONTROL_SUFFIX)
        for condition in args.conditions
    }
    if (
        selected_audio_conditions & PREDICTED_CONDITIONS
        and args.predictions_root is None
    ):
        parser.error("--predictions-root is required by the selected conditions")
    if (args.predictions_report is not None or args.predictions_mode is not None) and (
        args.predictions_root is None
    ):
        parser.error("prediction report/mode overrides require --predictions-root")
    local_model = Path(args.model).expanduser().exists()
    if not local_model and args.revision is None:
        parser.error("remote --model requires an explicit pinned --revision")
    return args


def _sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_audio_condition(condition: str) -> str:
    return condition.removesuffix(OPTION_ORDER_CONTROL_SUFFIX)


def presented_answer_options(
    record: AudioQARecord, condition: str, seed: int
) -> Tuple[str, ...]:
    """Return a gold-independent registered or option-only presentation.

    The non-zero cyclic shift is derived from only the public record ID and
    run seed.  It neither reads the answer nor its original position, yet it
    guarantees that every option (including the gold one) changes position.
    """

    options = tuple(record.answer_options)
    if not condition.endswith(OPTION_ORDER_CONTROL_SUFFIX):
        return options
    if len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("option-order control requires at least two unique options")
    digest = hashlib.sha256(
        f"qces-option-order-v1:{seed}:{record.sample_id}".encode("utf-8")
    ).digest()
    shift = 1 + int.from_bytes(digest[:8], "big") % (len(options) - 1)
    return options[shift:] + options[:shift]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_provenance() -> Dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _model_source_provenance(model_name: str, hash_weights: bool) -> Dict[str, Any]:
    path = Path(model_name).expanduser()
    if not path.exists():
        return {"kind": "huggingface_repository", "repository_id": model_name}
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"local --model must be a directory: {root}")
    small_names = {
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "special_tokens_map.json",
        "chat_template.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    metadata_hashes: Dict[str, str] = {}
    weight_inventory: List[Dict[str, Any]] = []
    for item in sorted(root.iterdir()):
        if not item.is_file():
            continue
        if item.name in small_names or item.suffix in {".jinja", ".model"}:
            metadata_hashes[item.name] = _sha256_file(item)
        if item.suffix in {".safetensors", ".bin"}:
            entry: Dict[str, Any] = {"name": item.name, "bytes": item.stat().st_size}
            if hash_weights:
                entry["sha256"] = _sha256_file(item)
            weight_inventory.append(entry)
    inferred_commit = None
    parts = root.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            inferred_commit = parts[index + 1]
    return {
        "kind": "local_huggingface_directory",
        "path": str(root),
        "inferred_snapshot_commit": inferred_commit,
        "metadata_file_sha256": metadata_hashes,
        "weight_inventory": weight_inventory,
        "weight_bytes_sha256_included": hash_weights,
        "metadata_inventory_sha256": _sha256_json(
            {"metadata": metadata_hashes, "weights": weight_inventory}
        ),
    }


def _parse_audioqa_record(payload: Dict[str, Any]) -> AudioQARecord:
    schema_version = payload.get("schema_version")
    if schema_version == "qces_v4":
        return parse_qces_v4_record(payload)
    if schema_version in QCES_V5_SCHEMA_VERSIONS:
        return parse_qces_v5_record(payload)
    raise ValueError(
        "AudioQA requires a qces_v4 or qces_v5 manifest; "
        f"got schema_version={schema_version!r}"
    )


def load_records(
    path: Path, split: str, max_records: Optional[int]
) -> List[AudioQARecord]:
    records = [_parse_audioqa_record(payload) for payload in read_jsonl(path)]
    if records and len({type(record) for record in records}) != 1:
        raise ValueError("mixed QCES schema families are not supported")
    if split != "all":
        records = [record for record in records if record.split == split]
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise ValueError("no QCES v4/v5 records remain after split/limit selection")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("manifest contains duplicate record IDs")
    return records


def _safe_manifest_audio(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"manifest audio path escapes dataset root: {relative}")
    return path


def _safe_child_path(root: Path, *parts: str) -> Path:
    """Resolve untrusted manifest-derived components below ``root``."""

    resolved_root = root.resolve()
    path = resolved_root.joinpath(*parts).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError("prediction path escapes predictions root: " + "/".join(parts))
    return path


def _prediction_path(root: Path, record: AudioQARecord, filename: str) -> Path:
    return _safe_child_path(
        root,
        record.scene_id,
        f"q{record.question_index}_{record.question_type}",
        filename,
    )


def _independent_scene_group(record: AudioQARecord) -> str:
    """Return the independence unit used by controls and uncertainty."""

    if isinstance(record, QCESV5Record):
        return record.scene_family_id
    return record.scene_id


def shuffled_record_map(
    records: Sequence[AudioQARecord],
) -> Dict[str, AudioQARecord]:
    """Map each question to a matched question from another independent scene.

    Counterfactual variants in QCES v5 share source audio and semantics, so a
    different ``scene_id`` is not enough: shuffled evidence must cross
    ``scene_family_id`` as well.
    """

    groups: Dict[Tuple[int, str], List[AudioQARecord]] = defaultdict(list)
    for record in records:
        groups[(record.question_index, record.question_type)].append(record)
    result: Dict[str, AudioQARecord] = {}
    for (question_index, question_type), group in groups.items():
        ordered = sorted(group, key=lambda item: (item.scene_id, item.sample_id))
        if len({_independent_scene_group(item) for item in ordered}) < 2:
            raise ValueError(
                "shuffled controls require at least two scenes from independent groups "
                "for every selected question; failed at "
                f"question_index={question_index}, question_type={question_type}"
            )
        for index, record in enumerate(ordered):
            source = next(
                ordered[(index + offset) % len(ordered)]
                for offset in range(1, len(ordered) + 1)
                if _independent_scene_group(ordered[(index + offset) % len(ordered)])
                != _independent_scene_group(record)
            )
            result[record.sample_id] = source
    return result


def build_input_descriptors(
    records: Sequence[AudioQARecord],
    manifest_root: Path,
    predictions_root: Optional[Path],
    conditions: Sequence[str],
) -> Dict[Tuple[str, str], InputDescriptor]:
    """Resolve and hash every input before the expensive model is loaded."""

    shuffled = (
        shuffled_record_map(records)
        if any(
            _base_audio_condition(value)
            in {"shuffled_evidence", "shuffled_oracle_evidence"}
            for value in conditions
        )
        else {}
    )
    file_hash_cache: Dict[Path, str] = {}

    def resolve_hashed_audio(
        relative: str, condition: str, record_id: str
    ) -> Tuple[Path, str]:
        path = _safe_manifest_audio(manifest_root, relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {condition} audio for {record_id}: {path}"
            )
        digest = file_hash_cache.get(path)
        if digest is None:
            digest = _sha256_file(path)
            file_hash_cache[path] = digest
        return path, digest

    def file_descriptor(
        record: AudioQARecord,
        condition: str,
        source: AudioQARecord,
        path: Path,
    ) -> InputDescriptor:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {condition} audio for {record.sample_id}: {path}"
            )
        digest = file_hash_cache.get(path)
        if digest is None:
            digest = _sha256_file(path)
            file_hash_cache[path] = digest
        return InputDescriptor(
            record_id=record.sample_id,
            condition=condition,
            source_record_id=source.sample_id,
            path=str(path),
            sha256=digest,
            sample_rate=source.sample_rate,
            num_samples=source.num_samples,
        )

    def derived_v5_oracle_descriptor(
        record: QCESV5Record,
        source: QCESV5Record,
        condition: str,
        audio_condition: str,
    ) -> InputDescriptor:
        if source.storage_mode != DERIVED_STORAGE_MODE:
            raise ValueError(
                "derived oracle requested for a materialized QCES v5 record"
            )
        mixture_path, mixture_sha256 = resolve_hashed_audio(
            source.mixture_path, condition, source.sample_id
        )
        event_map = {event.event_id: event for event in source.events}
        component_paths: List[str] = []
        component_sha256s: List[str] = []
        for event_id in source.evidence_event_ids:
            event_path, event_sha256 = resolve_hashed_audio(
                event_map[event_id].stem_path, condition, source.sample_id
            )
            component_paths.append(str(event_path))
            component_sha256s.append(event_sha256)
        recipe = (
            "sum(evidence_event_stems)"
            if audio_condition == "oracle_evidence"
            else "mixture-sum(evidence_event_stems)"
        )
        identity = {
            "recipe": recipe,
            "mixture_sha256": mixture_sha256,
            "evidence_event_ids": list(source.evidence_event_ids),
            "evidence_event_stem_sha256s": component_sha256s,
            "sample_rate": source.sample_rate,
            "num_samples": source.num_samples,
        }
        return InputDescriptor(
            record_id=record.sample_id,
            condition=condition,
            source_record_id=source.sample_id,
            path=str(mixture_path),
            sha256=_sha256_json(identity),
            sample_rate=source.sample_rate,
            num_samples=source.num_samples,
            special=f"derived_{audio_condition}",
            component_paths=tuple(component_paths),
            component_sha256s=tuple(component_sha256s),
            derivation_recipe=recipe,
        )

    descriptors: Dict[Tuple[str, str], InputDescriptor] = {}
    for record in records:
        for condition in conditions:
            key = (record.sample_id, condition)
            audio_condition = _base_audio_condition(condition)
            if audio_condition == "mixture":
                descriptors[key] = file_descriptor(
                    record,
                    condition,
                    record,
                    _safe_manifest_audio(manifest_root, record.mixture_path),
                )
            elif audio_condition == "oracle_evidence":
                if (
                    isinstance(record, QCESV5Record)
                    and record.storage_mode == DERIVED_STORAGE_MODE
                ):
                    descriptors[key] = derived_v5_oracle_descriptor(
                        record, record, condition, audio_condition
                    )
                else:
                    assert record.evidence_stem_path is not None
                    descriptors[key] = file_descriptor(
                        record,
                        condition,
                        record,
                        _safe_manifest_audio(manifest_root, record.evidence_stem_path),
                    )
            elif audio_condition == "oracle_residual":
                if (
                    isinstance(record, QCESV5Record)
                    and record.storage_mode == DERIVED_STORAGE_MODE
                ):
                    descriptors[key] = derived_v5_oracle_descriptor(
                        record, record, condition, audio_condition
                    )
                else:
                    assert record.residual_stem_path is not None
                    descriptors[key] = file_descriptor(
                        record,
                        condition,
                        record,
                        _safe_manifest_audio(manifest_root, record.residual_stem_path),
                    )
            elif audio_condition in {"predicted_evidence", "predicted_residual"}:
                assert predictions_root is not None
                filename = (
                    "predicted_evidence.wav"
                    if audio_condition == "predicted_evidence"
                    else "predicted_residual.wav"
                )
                descriptors[key] = file_descriptor(
                    record,
                    condition,
                    record,
                    _prediction_path(predictions_root, record, filename),
                )
            elif audio_condition == "shuffled_evidence":
                assert predictions_root is not None
                source = shuffled[record.sample_id]
                descriptors[key] = file_descriptor(
                    record,
                    condition,
                    source,
                    _prediction_path(
                        predictions_root, source, "predicted_evidence.wav"
                    ),
                )
            elif audio_condition == "shuffled_oracle_evidence":
                source = shuffled[record.sample_id]
                if (
                    isinstance(record, QCESV5Record)
                    and isinstance(source, QCESV5Record)
                    and source.storage_mode == DERIVED_STORAGE_MODE
                ):
                    descriptors[key] = derived_v5_oracle_descriptor(
                        record, source, condition, "oracle_evidence"
                    )
                else:
                    assert source.evidence_stem_path is not None
                    descriptors[key] = file_descriptor(
                        record,
                        condition,
                        source,
                        _safe_manifest_audio(manifest_root, source.evidence_stem_path),
                    )
            elif audio_condition == "silence":
                digest = hashlib.sha256(
                    f"float32-zero:{record.sample_rate}:{record.num_samples}".encode()
                ).hexdigest()
                descriptors[key] = InputDescriptor(
                    record_id=record.sample_id,
                    condition=condition,
                    source_record_id=record.sample_id,
                    path=None,
                    sha256=digest,
                    sample_rate=record.sample_rate,
                    num_samples=record.num_samples,
                    special="silence",
                )
            elif audio_condition == "question_only":
                descriptors[key] = InputDescriptor(
                    record_id=record.sample_id,
                    condition=condition,
                    source_record_id=record.sample_id,
                    path=None,
                    sha256=hashlib.sha256(b"question-only:no-audio").hexdigest(),
                    sample_rate=record.sample_rate,
                    num_samples=0,
                    special="question_only",
                )
            else:  # pragma: no cover - argparse guards this path
                raise ValueError(f"unsupported condition: {condition}")
    return descriptors


@lru_cache(maxsize=256)
def _read_audio_cached(path_string: str) -> Tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path_string, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        raise ValueError(
            f"expected mono audio at {path_string}, got {waveform.shape[1]} channels"
        )
    mono = np.ascontiguousarray(waveform[:, 0])
    if not np.isfinite(mono).all():
        raise ValueError(f"audio contains NaN/Inf: {path_string}")
    return mono, int(sample_rate)


def materialize_input(
    descriptor: InputDescriptor,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    if descriptor.special == "question_only":
        return None, None
    if descriptor.special == "silence":
        return (
            np.zeros(descriptor.num_samples, dtype=np.float32),
            descriptor.sample_rate,
        )
    if descriptor.special in {
        "derived_oracle_evidence",
        "derived_oracle_residual",
    }:
        evidence = np.zeros(descriptor.num_samples, dtype=np.float32)
        for component_path in descriptor.component_paths:
            component, component_rate = _read_audio_cached(component_path)
            if component_rate != descriptor.sample_rate:
                raise ValueError(
                    f"sample-rate mismatch at {component_path}: "
                    f"{component_rate} != {descriptor.sample_rate}"
                )
            if component.shape[0] != descriptor.num_samples:
                raise ValueError(
                    f"sample-count mismatch at {component_path}: "
                    f"{component.shape[0]} != {descriptor.num_samples}"
                )
            evidence += component
        if descriptor.special == "derived_oracle_evidence":
            return evidence, descriptor.sample_rate
        assert descriptor.path is not None
        mixture, mixture_rate = _read_audio_cached(descriptor.path)
        if mixture_rate != descriptor.sample_rate:
            raise ValueError(
                f"sample-rate mismatch at {descriptor.path}: "
                f"{mixture_rate} != {descriptor.sample_rate}"
            )
        if mixture.shape[0] != descriptor.num_samples:
            raise ValueError(
                f"sample-count mismatch at {descriptor.path}: "
                f"{mixture.shape[0]} != {descriptor.num_samples}"
            )
        return mixture.copy() - evidence, descriptor.sample_rate
    assert descriptor.path is not None
    waveform, sample_rate = _read_audio_cached(descriptor.path)
    if sample_rate != descriptor.sample_rate:
        raise ValueError(
            f"sample-rate mismatch at {descriptor.path}: {sample_rate} != {descriptor.sample_rate}"
        )
    if waveform.shape[0] != descriptor.num_samples:
        raise ValueError(
            f"sample-count mismatch at {descriptor.path}: {waveform.shape[0]} != {descriptor.num_samples}"
        )
    return waveform.copy(), sample_rate


def format_mc_prompt(question: str, options: Sequence[str]) -> str:
    if not 2 <= len(options) <= len(OPTION_LABELS):
        raise ValueError("AudioQA requires between two and five answer options")
    labels = OPTION_LABELS[: len(options)]
    rendered = []
    for label, option in zip(labels, options):
        display = (
            "no_evidence (the queried reference sound is absent)"
            if option == "no_evidence"
            else option
        )
        rendered.append(f"{label}. {display}")
    return (
        "Answer the multiple-choice question using only the supplied audio.\n"
        f"Question: {question}\n"
        "Options:\n"
        + "\n".join(rendered)
        + "\nRespond with exactly one option letter: "
        + ", ".join(labels)
        + ".\nAnswer:"
    )


def _softmax(values: Sequence[float]) -> Tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return tuple(float(item) for item in probabilities)


def encode_option_continuations(tokenizer: Any) -> Tuple[Tuple[int, ...], ...]:
    """Tokenize the exact assistant continuations requested by the prompt."""

    return tuple(
        tuple(tokenizer.encode(label, add_special_tokens=False))
        for label in OPTION_LABELS
    )


def reconcile_projector_output_dtype(output: Any, target_dtype: Any) -> Any:
    """Match projected audio embeddings to the language embedding dtype.

    Transformers 5.14 moves AF3 audio embeddings to the language-embedding
    *device* before ``masked_scatter`` but does not also match their dtype.  A
    mixed FP16/FP32 audio tower can therefore return Float projected features
    for Half language embeddings.  Keeping this workaround at the projector
    boundary avoids changing either the installed Transformers package or the
    model's numerical path elsewhere.
    """

    if not hasattr(output, "dtype") or not hasattr(output, "to"):
        raise TypeError("AF3 multi-modal projector must return a tensor")
    if output.dtype == target_dtype:
        return output
    return output.to(dtype=target_dtype)


class AudioFlamingo3OptionScorer:
    """Official Transformers AF3 option-log-probability scorer."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: Optional[str],
        quantization: str,
        dtype: str,
        device: str,
        device_map: str,
        attention_implementation: str,
        local_files_only: bool,
        seed: int,
    ) -> None:
        try:
            import torch
            from transformers import (
                AudioFlamingo3ForConditionalGeneration,
                AutoProcessor,
            )
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "Audio Flamingo 3 requires a recent Transformers release containing "
                "AudioFlamingo3ForConditionalGeneration."
            ) from exc

        self.torch = torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        dtype_value = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        if quantization != "none" and device not in {"auto", "cuda"}:
            raise ValueError("k-bit AF3 loading supports --device auto/cuda, not cpu")
        load_kwargs: Dict[str, Any] = {
            "local_files_only": local_files_only,
            "dtype": dtype_value,
            "attn_implementation": attention_implementation,
            "low_cpu_mem_usage": True,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        if quantization != "none":
            if (
                _package_version("bitsandbytes") is None
                or _package_version("accelerate") is None
            ):
                raise RuntimeError(
                    f"--quantization {quantization} requires both bitsandbytes and "
                    "accelerate in this environment"
                )
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype_value,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            load_kwargs["device_map"] = device_map

        processor_kwargs: Dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None:
            processor_kwargs["revision"] = revision
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs
        ).eval()
        if quantization == "none":
            resolved_device = (
                "cuda"
                if device == "auto" and torch.cuda.is_available()
                else "cpu" if device == "auto" else device
            )
            self.model.to(resolved_device)

        self.input_device = next(self.model.parameters()).device
        self.audio_dtype = next(self.model.model.audio_tower.parameters()).dtype
        self.compute_dtype = dtype_value
        language_embeddings = self.model.model.language_model.get_input_embeddings()
        self.language_embedding_dtype = language_embeddings.weight.dtype
        target_embedding_dtype = self.language_embedding_dtype

        def projector_dtype_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            return reconcile_projector_output_dtype(output, target_embedding_dtype)

        self._projector_dtype_hook = (
            self.model.model.multi_modal_projector.register_forward_hook(
                projector_dtype_hook
            )
        )
        self.target_sample_rate = int(self.processor.feature_extractor.sampling_rate)
        self._model_name = model_name
        self._revision = revision
        self._quantization = quantization
        self._dtype = dtype
        self._device_map = device_map if quantization != "none" else None
        self._attention_implementation = attention_implementation
        # The official AF3 chat template ends the generation prefix with
        # ``<|im_start|>assistant\n``.  Therefore the exact requested response
        # starts with ``A`` rather than `` A``.  Scoring the space-prefixed token
        # would evaluate a different continuation from the one the model is
        # instructed to emit.
        self._candidate_ids = encode_option_continuations(self.processor.tokenizer)
        if any(not token_ids for token_ids in self._candidate_ids):
            raise RuntimeError(
                "AF3 tokenizer produced an empty option-letter continuation"
            )

    def provenance(self) -> Mapping[str, Any]:
        config = self.model.config
        return {
            "auditor_family": "audio_flamingo_3",
            "model_argument": self._model_name,
            "requested_revision": self._revision,
            "resolved_commit_hash": getattr(config, "_commit_hash", None),
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "transformers_version": _package_version("transformers"),
            "torch_version": _package_version("torch"),
            "bitsandbytes_version": _package_version("bitsandbytes"),
            "accelerate_version": _package_version("accelerate"),
            "quantization": self._quantization,
            "dtype": self._dtype,
            "device_map": self._device_map,
            "input_device": str(self.input_device),
            "audio_tower_dtype": str(self.audio_dtype),
            "autocast_compute_dtype": str(self.compute_dtype),
            "language_embedding_dtype": str(self.language_embedding_dtype),
            "projector_output_dtype_reconciliation": True,
            "attention_implementation": self._attention_implementation,
            "processor_sample_rate": self.target_sample_rate,
            "option_continuation_token_ids": [
                list(item) for item in self._candidate_ids
            ],
            "option_continuation_text": list(OPTION_LABELS),
            "scoring_version": SCORING_VERSION,
        }

    def _resample(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == self.target_sample_rate:
            return np.ascontiguousarray(waveform, dtype=np.float32)
        divisor = math.gcd(sample_rate, self.target_sample_rate)
        result = resample_poly(
            waveform,
            self.target_sample_rate // divisor,
            sample_rate // divisor,
        )
        return np.ascontiguousarray(result, dtype=np.float32)

    def _prepare(
        self, prompt: str, waveform: Optional[np.ndarray], sample_rate: Optional[int]
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        kwargs: Dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if waveform is not None:
            if sample_rate is None:
                raise ValueError("sample_rate is required when waveform is provided")
            audio = self._resample(waveform, sample_rate)
            content.append({"type": "audio", "audio": audio})
            # Transformers >=5 routes processor-call arguments through this
            # explicit wrapper.  A top-level ``audio_kwargs`` is deprecated, and
            # a top-level/common ``padding=True`` overrides AF3's required
            # max-length Whisper padding, producing 400 frames for an 8 s clip
            # while the encoder positional table expects 1500.
            kwargs["processor_kwargs"] = {
                "audio_kwargs": {
                    "sampling_rate": self.target_sample_rate,
                    "return_attention_mask": True,
                    "padding": "max_length",
                }
            }
        conversation = [[{"role": "user", "content": content}]]
        prepared = self.processor.apply_chat_template(conversation, **kwargs)
        allowed = {
            "input_ids",
            "attention_mask",
            "input_features",
            "input_features_mask",
        }
        result: Dict[str, Any] = {}
        for key, value in prepared.items():
            if key not in allowed:
                continue
            value = value.to(self.input_device)
            result[key] = value
        return result

    def _model_forward_context(self) -> Any:
        """Autocast the mixed-precision audio tower without downcasting inputs."""

        if self.input_device.type == "cuda" and self.compute_dtype in {
            self.torch.float16,
            self.torch.bfloat16,
        }:
            return self.torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        return contextlib.nullcontext()

    def score(
        self,
        question: str,
        options: Sequence[str],
        waveform: Optional[np.ndarray],
        sample_rate: Optional[int],
    ) -> OptionScore:
        torch = self.torch
        prompt = format_mc_prompt(question, options)
        inputs = self._prepare(prompt, waveform, sample_rate)
        candidate_ids = self._candidate_ids[: len(options)]
        token_lengths = tuple(len(item) for item in candidate_ids)
        scores: List[float] = []
        # Keep processor log-Mel features in FP32.  In k-bit AF3 the frozen
        # audio tower can contain FP16 convolutions and FP32 LayerNorms; CUDA
        # autocast is what safely bridges those mixed parameter dtypes.
        with torch.inference_mode(), self._model_forward_context():
            if all(length == 1 for length in token_lengths):
                output = self.model(
                    **inputs, use_cache=False, logits_to_keep=1, return_dict=True
                )
                log_probs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
                scores = [
                    float(log_probs[token_ids[0]].cpu()) for token_ids in candidate_ids
                ]
                method = "single_token_next_log_probability"
            else:
                prefix_width = inputs["input_ids"].shape[1]
                for token_ids in candidate_ids:
                    continuation = torch.tensor(
                        [token_ids], dtype=torch.long, device=self.input_device
                    )
                    candidate_inputs = dict(inputs)
                    candidate_inputs["input_ids"] = torch.cat(
                        [inputs["input_ids"], continuation], dim=1
                    )
                    extension_mask = torch.ones_like(continuation)
                    candidate_inputs["attention_mask"] = torch.cat(
                        [inputs["attention_mask"], extension_mask], dim=1
                    )
                    output = self.model(
                        **candidate_inputs,
                        use_cache=False,
                        logits_to_keep=0,
                        return_dict=True,
                    )
                    log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
                    pieces = [
                        log_probs[prefix_width + offset - 1, token_id]
                        for offset, token_id in enumerate(token_ids)
                    ]
                    # This is the conditional log probability of the exact
                    # continuation sequence.  (The five ASCII letters are
                    # normally single tokens, but the fallback remains exact.)
                    scores.append(float(torch.stack(pieces).sum().cpu()))
                method = "sum_continuation_token_log_probability"
        probabilities = _softmax(scores)
        predicted = min(range(len(scores)), key=lambda index: (-scores[index], index))
        return OptionScore(
            log_scores=tuple(scores),
            probabilities=probabilities,
            predicted_index=predicted,
            scoring_method=method,
            candidate_token_lengths=token_lengths,
        )

    def generate_text(
        self,
        prompt: str,
        waveform: Optional[np.ndarray],
        sample_rate: Optional[int],
        *,
        max_new_tokens: int,
    ) -> str:
        """Deterministically generate one frozen-AF3 text continuation.

        This is used by the separately versioned caption/planner baseline.  It
        deliberately shares the exact audio preprocessing, dtype reconciliation,
        quantization, and model provenance of the multiple-choice auditor while
        never receiving an answer, answer index, or answer-option list.
        """

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        torch = self.torch
        inputs = self._prepare(prompt, waveform, sample_rate)
        prefix_width = inputs["input_ids"].shape[1]
        with torch.inference_mode(), self._model_forward_context():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=False,
            )
        continuation = generated[0, prefix_width:].detach().cpu().tolist()
        return self.processor.tokenizer.decode(
            continuation, skip_special_tokens=True
        ).strip()


class Qwen2AudioOptionScorer:
    """Official Transformers Qwen2-Audio exact-option scorer.

    The gold answer and gold option index never enter this class.  It receives
    only the condition waveform, question, and unmarked answer choices, then
    returns conditional log probabilities for the exact assistant
    continuations ``A`` through ``E``.  This makes its results directly
    comparable with the AF3 audit without relying on free-generation parsing.
    """

    def __init__(
        self,
        *,
        model_name: str,
        revision: Optional[str],
        quantization: str,
        dtype: str,
        device: str,
        device_map: str,
        attention_implementation: str,
        local_files_only: bool,
        seed: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "Qwen2-Audio requires a recent Transformers release containing "
                "Qwen2AudioForConditionalGeneration."
            ) from exc

        self.torch = torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        dtype_value = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        if quantization != "none" and device not in {"auto", "cuda"}:
            raise ValueError(
                "k-bit Qwen2-Audio loading supports --device auto/cuda, not cpu"
            )
        load_kwargs: Dict[str, Any] = {
            "local_files_only": local_files_only,
            "dtype": dtype_value,
            "attn_implementation": attention_implementation,
            "low_cpu_mem_usage": True,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        if quantization != "none":
            if (
                _package_version("bitsandbytes") is None
                or _package_version("accelerate") is None
            ):
                raise RuntimeError(
                    f"--quantization {quantization} requires both bitsandbytes and "
                    "accelerate in this environment"
                )
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype_value,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            load_kwargs["device_map"] = device_map

        processor_kwargs: Dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None:
            processor_kwargs["revision"] = revision
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs
        ).eval()
        if quantization == "none":
            resolved_device = (
                "cuda"
                if device == "auto" and torch.cuda.is_available()
                else "cpu" if device == "auto" else device
            )
            self.model.to(resolved_device)

        self.input_device = next(self.model.parameters()).device
        self.audio_dtype = next(self.model.model.audio_tower.parameters()).dtype
        self.compute_dtype = dtype_value
        self.language_embedding_dtype = (
            self.model.model.language_model.get_input_embeddings().weight.dtype
        )
        self.target_sample_rate = int(self.processor.feature_extractor.sampling_rate)
        self._model_name = model_name
        self._revision = revision
        self._quantization = quantization
        self._dtype = dtype
        self._device_map = device_map if quantization != "none" else None
        self._attention_implementation = attention_implementation
        self._candidate_ids = encode_option_continuations(self.processor.tokenizer)
        if any(not token_ids for token_ids in self._candidate_ids):
            raise RuntimeError(
                "Qwen2-Audio tokenizer produced an empty option-letter continuation"
            )

    def provenance(self) -> Mapping[str, Any]:
        config = self.model.config
        return {
            "auditor_family": "qwen2_audio",
            "model_argument": self._model_name,
            "requested_revision": self._revision,
            "resolved_commit_hash": getattr(config, "_commit_hash", None),
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "transformers_version": _package_version("transformers"),
            "torch_version": _package_version("torch"),
            "bitsandbytes_version": _package_version("bitsandbytes"),
            "accelerate_version": _package_version("accelerate"),
            "quantization": self._quantization,
            "dtype": self._dtype,
            "device_map": self._device_map,
            "input_device": str(self.input_device),
            "audio_tower_dtype": str(self.audio_dtype),
            "autocast_compute_dtype": str(self.compute_dtype),
            "language_embedding_dtype": str(self.language_embedding_dtype),
            "attention_implementation": self._attention_implementation,
            "processor_sample_rate": self.target_sample_rate,
            "option_continuation_token_ids": [
                list(item) for item in self._candidate_ids
            ],
            "option_continuation_text": list(OPTION_LABELS),
            "scoring_version": SCORING_VERSION,
        }

    def _resample(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == self.target_sample_rate:
            return np.ascontiguousarray(waveform, dtype=np.float32)
        divisor = math.gcd(sample_rate, self.target_sample_rate)
        result = resample_poly(
            waveform,
            self.target_sample_rate // divisor,
            sample_rate // divisor,
        )
        return np.ascontiguousarray(result, dtype=np.float32)

    def _prepare(
        self, prompt: str, waveform: Optional[np.ndarray], sample_rate: Optional[int]
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        audio: Optional[np.ndarray] = None
        if waveform is not None:
            if sample_rate is None:
                raise ValueError("sample_rate is required when waveform is provided")
            audio = self._resample(waveform, sample_rate)
            # The in-memory value is only a template marker.  The waveform is
            # supplied independently to the processor, exactly as in the
            # official Qwen2-Audio audio-analysis recipe.
            content.append({"type": "audio", "audio": "in_memory"})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        processor_kwargs: Dict[str, Any] = {
            "text": text_prompt,
            "return_tensors": "pt",
            "padding": True,
        }
        if audio is not None:
            processor_kwargs["audio"] = audio
            processor_kwargs["sampling_rate"] = self.target_sample_rate
        prepared = self.processor(**processor_kwargs)
        allowed = {
            "input_ids",
            "attention_mask",
            "input_features",
            "feature_attention_mask",
        }
        return {
            key: value.to(self.input_device)
            for key, value in prepared.items()
            if key in allowed
        }

    def _model_forward_context(self) -> Any:
        if self.input_device.type == "cuda" and self.compute_dtype in {
            self.torch.float16,
            self.torch.bfloat16,
        }:
            return self.torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        return contextlib.nullcontext()

    def score(
        self,
        question: str,
        options: Sequence[str],
        waveform: Optional[np.ndarray],
        sample_rate: Optional[int],
    ) -> OptionScore:
        torch = self.torch
        inputs = self._prepare(
            format_mc_prompt(question, options), waveform, sample_rate
        )
        candidate_ids = self._candidate_ids[: len(options)]
        token_lengths = tuple(len(item) for item in candidate_ids)
        scores: List[float] = []
        with torch.inference_mode(), self._model_forward_context():
            if all(length == 1 for length in token_lengths):
                output = self.model(**inputs, use_cache=False, return_dict=True)
                log_probs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
                scores = [
                    float(log_probs[token_ids[0]].cpu()) for token_ids in candidate_ids
                ]
                method = "single_token_next_log_probability"
            else:
                prefix_width = inputs["input_ids"].shape[1]
                for token_ids in candidate_ids:
                    continuation = torch.tensor(
                        [token_ids], dtype=torch.long, device=self.input_device
                    )
                    candidate_inputs = dict(inputs)
                    candidate_inputs["input_ids"] = torch.cat(
                        [inputs["input_ids"], continuation], dim=1
                    )
                    candidate_inputs["attention_mask"] = torch.cat(
                        [inputs["attention_mask"], torch.ones_like(continuation)],
                        dim=1,
                    )
                    output = self.model(
                        **candidate_inputs, use_cache=False, return_dict=True
                    )
                    log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
                    pieces = [
                        log_probs[prefix_width + offset - 1, token_id]
                        for offset, token_id in enumerate(token_ids)
                    ]
                    scores.append(float(torch.stack(pieces).sum().cpu()))
                method = "sum_continuation_token_log_probability"
        probabilities = _softmax(scores)
        predicted = min(range(len(scores)), key=lambda index: (-scores[index], index))
        return OptionScore(
            log_scores=tuple(scores),
            probabilities=probabilities,
            predicted_index=predicted,
            scoring_method=method,
            candidate_token_lengths=token_lengths,
        )


class Phi4MMOptionScorer:
    """Pinned Microsoft Phi-4 Multimodal exact-option scorer.

    Phi-4 MM uses a Phi language backbone and a Conformer speech encoder, so it
    provides a materially different held-out architecture from AF3/Qwen2-Audio.
    Its trusted custom code is loaded only from an explicitly pinned revision.
    """

    def __init__(
        self,
        *,
        model_name: str,
        revision: Optional[str],
        quantization: str,
        dtype: str,
        device: str,
        device_map: str,
        attention_implementation: str,
        local_files_only: bool,
        seed: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "Phi-4 Multimodal requires its pinned Transformers/PEFT runtime."
            ) from exc
        if not revision and not Path(model_name).expanduser().exists():
            raise ValueError("remote Phi-4 MM custom code requires a pinned revision")

        self.torch = torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        dtype_value = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        if quantization != "none" and device not in {"auto", "cuda"}:
            raise ValueError("k-bit Phi-4 MM loading supports --device auto/cuda")
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "torch_dtype": dtype_value,
            "_attn_implementation": attention_implementation,
            "low_cpu_mem_usage": True,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        if quantization != "none":
            if (
                _package_version("bitsandbytes") is None
                or _package_version("accelerate") is None
                or _package_version("peft") is None
            ):
                raise RuntimeError(
                    f"--quantization {quantization} for Phi-4 MM requires "
                    "bitsandbytes, accelerate, and peft"
                )
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype_value,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            load_kwargs["device_map"] = device_map

        processor_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        if revision is not None:
            processor_kwargs["revision"] = revision
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        ).eval()
        if getattr(self.model.config, "model_type", None) != "phi4mm":
            raise RuntimeError("selected Phi auditor did not resolve a phi4mm config")
        resolved_commit = getattr(self.model.config, "_commit_hash", None)
        if revision is not None and not Path(model_name).expanduser().exists():
            if resolved_commit != revision:
                raise RuntimeError(
                    "Phi-4 MM resolved commit does not match the requested pin: "
                    f"{resolved_commit} != {revision}"
                )
        if quantization == "none":
            resolved_device = (
                "cuda"
                if device == "auto" and torch.cuda.is_available()
                else "cpu" if device == "auto" else device
            )
            self.model.to(resolved_device)

        self.input_device = next(self.model.parameters()).device
        self.compute_dtype = dtype_value
        self.audio_dtype = next(
            self.model.model.embed_tokens_extend.audio_embed.parameters()
        ).dtype
        self.language_embedding_dtype = self.model.get_input_embeddings().weight.dtype
        self.target_sample_rate = 16_000
        self._model_name = model_name
        self._revision = revision
        self._quantization = quantization
        self._dtype = dtype
        self._device_map = device_map if quantization != "none" else None
        self._attention_implementation = attention_implementation
        self._candidate_ids = encode_option_continuations(self.processor.tokenizer)
        if any(not token_ids for token_ids in self._candidate_ids):
            raise RuntimeError("Phi-4 MM tokenizer produced an empty option letter")

    def provenance(self) -> Mapping[str, Any]:
        config = self.model.config
        return {
            "auditor_family": "phi4_multimodal",
            "architecture_independence_role": (
                "Phi language backbone plus 24-block Conformer speech encoder"
            ),
            "model_argument": self._model_name,
            "requested_revision": self._revision,
            "resolved_commit_hash": getattr(config, "_commit_hash", None),
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "trusted_custom_code": True,
            "transformers_version": _package_version("transformers"),
            "torch_version": _package_version("torch"),
            "torchvision_version": _package_version("torchvision"),
            "peft_version": _package_version("peft"),
            "bitsandbytes_version": _package_version("bitsandbytes"),
            "accelerate_version": _package_version("accelerate"),
            "quantization": self._quantization,
            "dtype": self._dtype,
            "device_map": self._device_map,
            "input_device": str(self.input_device),
            "audio_encoder_dtype": str(self.audio_dtype),
            "autocast_compute_dtype": str(self.compute_dtype),
            "language_embedding_dtype": str(self.language_embedding_dtype),
            "attention_implementation": self._attention_implementation,
            "processor_sample_rate": self.target_sample_rate,
            "option_continuation_token_ids": [
                list(item) for item in self._candidate_ids
            ],
            "option_continuation_text": list(OPTION_LABELS),
            "scoring_version": SCORING_VERSION,
        }

    def _resample(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == self.target_sample_rate:
            return np.ascontiguousarray(waveform, dtype=np.float32)
        divisor = math.gcd(sample_rate, self.target_sample_rate)
        result = resample_poly(
            waveform,
            self.target_sample_rate // divisor,
            sample_rate // divisor,
        )
        return np.ascontiguousarray(result, dtype=np.float32)

    def _prepare(
        self, prompt: str, waveform: Optional[np.ndarray], sample_rate: Optional[int]
    ) -> Dict[str, Any]:
        audio_marker = ""
        processor_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
        if waveform is not None:
            if sample_rate is None:
                raise ValueError("sample_rate is required when waveform is provided")
            audio = self._resample(waveform, sample_rate)
            audio_marker = "<|audio_1|>"
            processor_kwargs["audios"] = [(audio, self.target_sample_rate)]
        processor_kwargs["text"] = f"<|user|>{audio_marker}{prompt}<|end|><|assistant|>"
        prepared = self.processor(**processor_kwargs)
        result: Dict[str, Any] = {}
        for key, value in prepared.items():
            if value is None:
                continue
            if hasattr(value, "to"):
                value = value.to(self.input_device)
            result[key] = value
        return result

    def _model_forward_context(self) -> Any:
        if self.input_device.type == "cuda" and self.compute_dtype in {
            self.torch.float16,
            self.torch.bfloat16,
        }:
            return self.torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        return contextlib.nullcontext()

    def score(
        self,
        question: str,
        options: Sequence[str],
        waveform: Optional[np.ndarray],
        sample_rate: Optional[int],
    ) -> OptionScore:
        torch = self.torch
        inputs = self._prepare(
            format_mc_prompt(question, options), waveform, sample_rate
        )
        candidate_ids = self._candidate_ids[: len(options)]
        token_lengths = tuple(len(item) for item in candidate_ids)
        scores: List[float] = []
        with torch.inference_mode(), self._model_forward_context():
            if all(length == 1 for length in token_lengths):
                output = self.model(
                    **inputs,
                    use_cache=False,
                    num_logits_to_keep=1,
                    return_dict=True,
                )
                log_probs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
                scores = [
                    float(log_probs[token_ids[0]].cpu()) for token_ids in candidate_ids
                ]
                method = "single_token_next_log_probability"
            else:
                prefix_width = inputs["input_ids"].shape[1]
                for token_ids in candidate_ids:
                    continuation = torch.tensor(
                        [token_ids], dtype=torch.long, device=self.input_device
                    )
                    candidate_inputs = dict(inputs)
                    candidate_inputs["input_ids"] = torch.cat(
                        [inputs["input_ids"], continuation], dim=1
                    )
                    candidate_inputs["attention_mask"] = torch.cat(
                        [inputs["attention_mask"], torch.ones_like(continuation)],
                        dim=1,
                    )
                    output = self.model(
                        **candidate_inputs,
                        use_cache=False,
                        num_logits_to_keep=0,
                        return_dict=True,
                    )
                    log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
                    pieces = [
                        log_probs[prefix_width + offset - 1, token_id]
                        for offset, token_id in enumerate(token_ids)
                    ]
                    scores.append(float(torch.stack(pieces).sum().cpu()))
                method = "sum_continuation_token_log_probability"
        probabilities = _softmax(scores)
        predicted = min(range(len(scores)), key=lambda index: (-scores[index], index))
        return OptionScore(
            log_scores=tuple(scores),
            probabilities=probabilities,
            predicted_index=predicted,
            scoring_method=method,
            candidate_token_lengths=token_lengths,
        )


def score_record_without_gold_inputs(
    scorer: OptionScorer,
    record: AudioQARecord,
    waveform: Optional[np.ndarray],
    sample_rate: Optional[int],
    answer_options: Optional[Sequence[str]] = None,
) -> OptionScore:
    """Call the frozen auditor without exposing the oracle answer or its index."""

    return scorer.score(
        record.question,
        record.answer_options if answer_options is None else answer_options,
        waveform,
        sample_rate,
    )


def make_item(
    record: AudioQARecord,
    descriptor: InputDescriptor,
    score: OptionScore,
    run_fingerprint: str,
    answer_options: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    presented_options = tuple(
        record.answer_options if answer_options is None else answer_options
    )
    if len(set(presented_options)) != len(presented_options):
        raise ValueError("presented answer options must be unique")
    if set(presented_options) != set(record.answer_options):
        raise ValueError("presented answer options changed the registered option set")
    option_count = len(presented_options)
    if not (
        len(score.log_scores)
        == len(score.probabilities)
        == len(score.candidate_token_lengths)
        == option_count
    ):
        raise ValueError("scorer returned the wrong number of option values")
    if not all(math.isfinite(float(value)) for value in score.log_scores):
        raise ValueError("scorer returned a non-finite option log score")
    if not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in score.probabilities
    ) or not math.isclose(sum(score.probabilities), 1.0, abs_tol=1e-5):
        raise ValueError("scorer returned invalid option probabilities")
    if any(length <= 0 for length in score.candidate_token_lengths):
        raise ValueError("scorer returned an invalid candidate token length")
    if not 0 <= score.predicted_index < len(presented_options):
        raise ValueError("scorer predicted an invalid option index")
    expected_prediction = min(
        range(option_count),
        key=lambda index: (-score.log_scores[index], index),
    )
    if score.predicted_index != expected_prediction:
        raise ValueError("predicted option index disagrees with option log scores")
    predicted_answer = presented_options[score.predicted_index]
    gold_option_index = presented_options.index(record.answer)
    item = {
        "format": "qces_audioqa_item_v2",
        "run_fingerprint": run_fingerprint,
        "id": record.sample_id,
        "scene_id": record.scene_id,
        "split": record.split,
        "question_index": record.question_index,
        "question_type": record.question_type,
        "relation": record.relation,
        "question": record.question,
        "answer_options": list(presented_options),
        "gold_answer": record.answer,
        "gold_option_index": gold_option_index,
        "registered_gold_option_index": record.answer_option_index,
        "option_order_control": descriptor.condition.endswith(
            OPTION_ORDER_CONTROL_SUFFIX
        ),
        "no_evidence": record.no_evidence,
        "condition": descriptor.condition,
        "source_record_id": descriptor.source_record_id,
        "source_audio_path": descriptor.path,
        "source_audio_sha256": descriptor.sha256,
        "source_audio_identity_kind": (
            "derived_recipe_and_input_hashes"
            if descriptor.derivation_recipe is not None
            else "audio_file_bytes"
        ),
        "source_audio_derivation_recipe": descriptor.derivation_recipe,
        "source_audio_component_paths": list(descriptor.component_paths),
        "source_audio_component_sha256s": list(descriptor.component_sha256s),
        "predicted_option_index": score.predicted_index,
        "predicted_answer": predicted_answer,
        "correct": predicted_answer == record.answer,
        "option_log_scores": list(score.log_scores),
        "option_probabilities": list(score.probabilities),
        "gold_log_score": score.log_scores[gold_option_index],
        "gold_option_probability": score.probabilities[gold_option_index],
        "scoring_method": score.scoring_method,
        "candidate_token_lengths": list(score.candidate_token_lengths),
    }
    if isinstance(record, QCESV5Record):
        same_role_label = None
        if not record.no_evidence:
            anchor_labels = tuple(
                record.event_by_id(event_id).label
                for event_id in record.anchor_event_ids
            )
            answer_labels = tuple(
                record.event_by_id(event_id).label
                for event_id in record.answer_event_ids
            )
            same_role_label = anchor_labels == answer_labels
        item.update(
            {
                "scene_family_id": record.scene_family_id,
                "variant_id": record.variant_id,
                "counterfactual_group_id": record.counterfactual_group_id,
                "question_semantics_id": record.question_semantics_id,
                "primary_counterfactual_probe": record.primary_counterfactual_probe,
                "same_role_label": same_role_label,
                "same_label_repeat": record.same_label_repeat,
                "semantic_overlap": record.semantic_overlap,
                "max_polyphony": record.max_polyphony,
                "hard_case_tags": list(record.hard_case_tags),
                "surface_control_group_id": record.surface_control_group_id,
                "mention_order_variant": record.mention_order_variant,
            }
        )
    return item


def _mean(values: Iterable[float]) -> Optional[float]:
    materialized = list(values)
    return float(sum(materialized) / len(materialized)) if materialized else None


def _first_question_candidates(
    item: Mapping[str, Any],
) -> Optional[Tuple[str, str]]:
    """Return the two named candidates in textual order for a first question.

    QCES item files intentionally retain the rendered question and answer
    options.  Matching the option strings back into the question avoids any
    dependence on builder-only metadata and supports all current paraphrases.
    Ambiguous/malformed matches are excluded from shortcut diagnostics instead
    of guessing an order.
    """

    if item.get("relation") != "first":
        return None
    question = item.get("question")
    options = item.get("answer_options")
    if (
        not isinstance(question, str)
        or not isinstance(options, Sequence)
        or isinstance(options, (str, bytes))
    ):
        return None
    folded_question = question.casefold()
    mentions: List[Tuple[int, int, str]] = []
    for option_index, raw_option in enumerate(options):
        if not isinstance(raw_option, str) or raw_option == "no_evidence":
            continue
        position = folded_question.find(raw_option.casefold())
        if position >= 0:
            mentions.append((position, option_index, raw_option))
    if len(mentions) != 2:
        return None
    mentions.sort(key=lambda value: (value[0], value[1]))
    if mentions[0][0] == mentions[1][0]:
        return None
    return mentions[0][2], mentions[1][2]


def _first_question_shortcut_metrics(
    items: Sequence[Mapping[str, Any]],
) -> Dict[str, Optional[float]]:
    eligible: List[Tuple[Mapping[str, Any], Tuple[str, str]]] = []
    for item in items:
        candidates = _first_question_candidates(item)
        if candidates is not None and item.get("gold_answer") in candidates:
            eligible.append((item, candidates))

    named_predictions = [
        (item, candidates)
        for item, candidates in eligible
        if item.get("predicted_answer") in candidates
    ]
    predicted_first_rate = _mean(
        float(item.get("predicted_answer") == candidates[0])
        for item, candidates in named_predictions
    )
    gold_first_rate = _mean(
        float(item.get("gold_answer") == candidates[0]) for item, candidates in eligible
    )
    first_gold = [
        item
        for item, candidates in eligible
        if item.get("gold_answer") == candidates[0]
    ]
    second_gold = [
        item
        for item, candidates in eligible
        if item.get("gold_answer") == candidates[1]
    ]
    first_accuracy = _mean(float(item["correct"]) for item in first_gold)
    second_accuracy = _mean(float(item["correct"]) for item in second_gold)
    return {
        "first_question_named_candidate_prediction_rate_↑": _mean(
            float(item.get("predicted_answer") in candidates)
            for item, candidates in eligible
        ),
        "first_question_first_mention_bias_gap_↓": (
            None
            if predicted_first_rate is None or gold_first_rate is None
            else abs(predicted_first_rate - gold_first_rate)
        ),
        "first_question_mention_position_accuracy_gap_↓": (
            None
            if first_accuracy is None or second_accuracy is None
            else abs(first_accuracy - second_accuracy)
        ),
    }


def _surface_control_pair_summary(
    items: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Optional[float]], Dict[str, int]]:
    """Measure invariance on pre-registered QCES-v5 first-question pairs.

    Each complete pair reverses the two candidates in the question and also
    independently permutes the five answer-option positions.  Consequently,
    these metrics test the *combined* surface intervention; they must not be
    described as an isolated option-order ablation.

    A one-sided group is counted as incomplete rather than rejected because
    ``--max-records`` can legitimately truncate an evaluation slice.  Complete
    but structurally inconsistent pairs are rejected to prevent a dataset bug
    from being reported as model robustness.
    """

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    eligible_rows = 0
    for item in items:
        group_id = item.get("surface_control_group_id")
        if group_id is None:
            continue
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("surface-control group ID must be a non-empty string")
        eligible_rows += 1
        grouped[group_id].append(item)

    pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    incomplete_groups = 0
    for group_id, group_items in grouped.items():
        if len(group_items) == 1:
            incomplete_groups += 1
            continue
        if len(group_items) != 2:
            raise ValueError(
                f"surface-control group {group_id!r} has {len(group_items)} rows, expected 2"
            )
        by_variant = {
            str(item.get("mention_order_variant")): item for item in group_items
        }
        if set(by_variant) != {"forward", "reversed"}:
            raise ValueError(
                f"surface-control group {group_id!r} lacks forward/reversed variants"
            )
        forward = by_variant["forward"]
        reversed_item = by_variant["reversed"]
        if (
            forward.get("relation") != "first"
            or reversed_item.get("relation") != "first"
        ):
            raise ValueError(
                f"surface-control group {group_id!r} contains a non-first question"
            )
        if forward.get("gold_answer") != reversed_item.get("gold_answer"):
            raise ValueError(
                f"surface-control group {group_id!r} changes the gold answer"
            )
        forward_options = forward.get("answer_options")
        reversed_options = reversed_item.get("answer_options")
        if (
            not isinstance(forward_options, Sequence)
            or isinstance(forward_options, (str, bytes))
            or not isinstance(reversed_options, Sequence)
            or isinstance(reversed_options, (str, bytes))
            or sorted(str(value) for value in forward_options)
            != sorted(str(value) for value in reversed_options)
        ):
            raise ValueError(
                f"surface-control group {group_id!r} changes the answer-option set"
            )
        pairs.append((forward, reversed_item))

    metrics: Dict[str, Optional[float]] = {}
    if pairs:
        metrics = {
            "first_question_surface_pair_prediction_invariance_↑": _mean(
                float(
                    forward.get("predicted_answer")
                    == reversed_item.get("predicted_answer")
                )
                for forward, reversed_item in pairs
            ),
            "first_question_surface_pair_both_correct_rate_↑": _mean(
                float(bool(forward["correct"]) and bool(reversed_item["correct"]))
                for forward, reversed_item in pairs
            ),
            "first_question_surface_pair_at_least_one_correct_rate_↑": _mean(
                float(bool(forward["correct"]) or bool(reversed_item["correct"]))
                for forward, reversed_item in pairs
            ),
            "first_question_surface_pair_gold_log_score_absolute_gap_↓": _mean(
                abs(
                    float(forward["gold_log_score"])
                    - float(reversed_item["gold_log_score"])
                )
                for forward, reversed_item in pairs
            ),
            "first_question_surface_pair_gold_probability_absolute_gap_↓": _mean(
                abs(
                    float(forward["gold_option_probability"])
                    - float(reversed_item["gold_option_probability"])
                )
                for forward, reversed_item in pairs
            ),
        }
    coverage = {
        "eligible_rows": eligible_rows,
        "complete_pairs": len(pairs),
        "incomplete_groups": incomplete_groups,
    }
    return metrics, coverage


def condition_metrics(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("cannot summarize an empty condition")
    answerable = [item for item in items if not item["no_evidence"]]
    negatives = [item for item in items if item["no_evidence"]]
    no_evidence_recall = _mean(float(item["correct"]) for item in negatives)
    answerable_recall = _mean(
        float(item["predicted_answer"] != "no_evidence") for item in answerable
    )
    balanced = (
        None
        if no_evidence_recall is None or answerable_recall is None
        else (no_evidence_recall + answerable_recall) / 2.0
    )
    result = {
        "multiple_choice_accuracy_all_↑": _mean(
            float(item["correct"]) for item in items
        ),
        "answerable_accuracy_↑": _mean(float(item["correct"]) for item in answerable),
        "no_evidence_accuracy_↑": no_evidence_recall,
        "answerability_balanced_accuracy_↑": balanced,
        "no_evidence_false_positive_rate_on_answerable_↓": (
            None if answerable_recall is None else 1.0 - answerable_recall
        ),
        "false_answer_rate_on_no_evidence_↓": (
            None if no_evidence_recall is None else 1.0 - no_evidence_recall
        ),
        "mean_gold_option_probability_↑": _mean(
            float(item["gold_option_probability"]) for item in items
        ),
        "candidate_aware_chance_accuracy_↑": _mean(
            0.5 if item["relation"] == "first" else 0.2 for item in items
        ),
        "answerable_candidate_aware_chance_accuracy_↑": _mean(
            0.5 if item["relation"] == "first" else 0.2 for item in answerable
        ),
        "accuracy_by_relation_↑": {
            relation: _mean(
                float(item["correct"]) for item in items if item["relation"] == relation
            )
            for relation in ("after", "before", "first")
        },
        "answerable_accuracy_by_relation_↑": {
            relation: _mean(
                float(item["correct"])
                for item in answerable
                if item["relation"] == relation
            )
            for relation in ("after", "before", "first")
        },
        "no_evidence_accuracy_by_relation_↑": {
            relation: _mean(
                float(item["correct"])
                for item in negatives
                if item["relation"] == relation
            )
            for relation in ("after", "before", "first")
        },
    }
    result.update(_first_question_shortcut_metrics(items))
    surface_metrics, _ = _surface_control_pair_summary(items)
    result.update(surface_metrics)
    return result


def _index_items(
    items: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    indexed: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item in items:
        condition = str(item["condition"])
        record_id = str(item["id"])
        if record_id in indexed[condition]:
            raise ValueError(
                f"duplicate item for condition/id: {(condition, record_id)}"
            )
        indexed[condition][record_id] = item
    return indexed


def option_order_control_summary(
    items: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, Dict[str, int]]]:
    """Summarize isolated answer-option permutations by acoustic condition."""

    indexed = _index_items(items)
    summaries: Dict[str, Dict[str, Optional[float]]] = {}
    coverage: Dict[str, Dict[str, int]] = {}
    control_conditions = sorted(
        condition
        for condition in indexed
        if condition.endswith(OPTION_ORDER_CONTROL_SUFFIX)
    )
    for control_condition in control_conditions:
        base_condition = _base_audio_condition(control_condition)
        if base_condition not in indexed:
            raise ValueError(
                f"option-order control lacks base condition: {base_condition}"
            )
        record_ids = sorted(
            set(indexed[base_condition]) & set(indexed[control_condition])
        )
        if len(record_ids) != len(indexed[control_condition]):
            raise ValueError(
                f"option-order control {control_condition} has unpaired records"
            )
        pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for record_id in record_ids:
            registered = indexed[base_condition][record_id]
            permuted = indexed[control_condition][record_id]
            if registered.get("question") != permuted.get("question"):
                raise ValueError("option-order control changed question text")
            if registered.get("source_audio_sha256") != permuted.get(
                "source_audio_sha256"
            ):
                raise ValueError("option-order control changed audio identity")
            registered_options = registered.get("answer_options")
            permuted_options = permuted.get("answer_options")
            if (
                not isinstance(registered_options, Sequence)
                or isinstance(registered_options, (str, bytes))
                or not isinstance(permuted_options, Sequence)
                or isinstance(permuted_options, (str, bytes))
                or sorted(str(value) for value in registered_options)
                != sorted(str(value) for value in permuted_options)
            ):
                raise ValueError("option-order control changed the answer-option set")
            if list(registered_options) == list(permuted_options):
                raise ValueError("option-order control did not change option order")
            if registered.get("gold_answer") != permuted.get("gold_answer"):
                raise ValueError("option-order control changed the gold answer")
            pairs.append((registered, permuted))

        summaries[base_condition] = {
            "option_order_gold_position_changed_rate_↑": _mean(
                float(
                    int(registered["gold_option_index"])
                    != int(permuted["gold_option_index"])
                )
                for registered, permuted in pairs
            ),
            "option_order_semantic_prediction_invariance_↑": _mean(
                float(
                    registered.get("predicted_answer")
                    == permuted.get("predicted_answer")
                )
                for registered, permuted in pairs
            ),
            "option_order_both_correct_rate_↑": _mean(
                float(bool(registered["correct"]) and bool(permuted["correct"]))
                for registered, permuted in pairs
            ),
            "option_order_at_least_one_correct_rate_↑": _mean(
                float(bool(registered["correct"]) or bool(permuted["correct"]))
                for registered, permuted in pairs
            ),
            "option_order_accuracy_absolute_gap_↓": (
                abs(
                    float(_mean(float(item["correct"]) for item, _ in pairs) or 0.0)
                    - float(_mean(float(item["correct"]) for _, item in pairs) or 0.0)
                )
                if pairs
                else None
            ),
            "option_order_gold_log_score_absolute_gap_↓": _mean(
                abs(
                    float(registered["gold_log_score"])
                    - float(permuted["gold_log_score"])
                )
                for registered, permuted in pairs
            ),
            "option_order_gold_probability_absolute_gap_↓": _mean(
                abs(
                    float(registered["gold_option_probability"])
                    - float(permuted["gold_option_probability"])
                )
                for registered, permuted in pairs
            ),
        }
        coverage[base_condition] = {
            "registered_records": len(indexed[base_condition]),
            "control_records": len(indexed[control_condition]),
            "complete_pairs": len(pairs),
        }
    return summaries, coverage


def paired_metrics(items: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    indexed = _index_items(items)
    answerable_ids = sorted(
        {str(item["id"]) for item in items if not item["no_evidence"]}
    )
    negative_ids = sorted({str(item["id"]) for item in items if item["no_evidence"]})

    def present(*conditions: str) -> List[str]:
        return [
            record_id
            for record_id in answerable_ids
            if all(record_id in indexed.get(condition, {}) for condition in conditions)
        ]

    def accuracy(condition: str, ids: Sequence[str]) -> Optional[float]:
        return _mean(
            float(indexed[condition][record_id]["correct"]) for record_id in ids
        )

    result: Dict[str, Optional[float]] = {}
    audio_dependent_ids: List[str] = []
    if "mixture" in indexed and "question_only" in indexed:
        ids = present("mixture", "question_only")
        audio_dependent_ids = [
            record_id
            for record_id in ids
            if indexed["mixture"][record_id]["correct"]
            and not indexed["question_only"][record_id]["correct"]
        ]
    if "predicted_evidence" in indexed:
        ids = present("predicted_evidence")
        result["predicted_evidence_sufficiency_accuracy_↑"] = accuracy(
            "predicted_evidence", ids
        )
        result["predicted_evidence_no_evidence_accuracy_↑"] = accuracy(
            "predicted_evidence",
            [
                record_id
                for record_id in negative_ids
                if record_id in indexed["predicted_evidence"]
            ],
        )
    if "predicted_residual" in indexed:
        ids = present("predicted_residual")
        result["predicted_residual_answer_leakage_accuracy_↓"] = accuracy(
            "predicted_residual", ids
        )
        result["predicted_residual_no_evidence_accuracy_↑"] = accuracy(
            "predicted_residual",
            [
                record_id
                for record_id in negative_ids
                if record_id in indexed["predicted_residual"]
            ],
        )
    if "oracle_evidence" in indexed:
        ids = present("oracle_evidence")
        result["oracle_evidence_sufficiency_accuracy_↑"] = accuracy(
            "oracle_evidence", ids
        )
        result["oracle_evidence_no_evidence_accuracy_↑"] = accuracy(
            "oracle_evidence",
            [
                record_id
                for record_id in negative_ids
                if record_id in indexed["oracle_evidence"]
            ],
        )
    if "oracle_residual" in indexed:
        ids = present("oracle_residual")
        result["oracle_residual_answer_leakage_accuracy_↓"] = accuracy(
            "oracle_residual", ids
        )
        result["oracle_residual_no_evidence_accuracy_↑"] = accuracy(
            "oracle_residual",
            [
                record_id
                for record_id in negative_ids
                if record_id in indexed["oracle_residual"]
            ],
        )

    if "mixture" in indexed and "question_only" in indexed:
        ids = present("mixture", "question_only")
        result["mixture_accuracy_gain_over_question_only_↑"] = _mean(
            float(indexed["mixture"][record_id]["correct"])
            - float(indexed["question_only"][record_id]["correct"])
            for record_id in ids
        )
        result["mixture_gold_log_score_gain_over_question_only_↑"] = _mean(
            float(indexed["mixture"][record_id]["gold_log_score"])
            - float(indexed["question_only"][record_id]["gold_log_score"])
            for record_id in ids
        )

    if "oracle_evidence" in indexed and "question_only" in indexed:
        ids = present("oracle_evidence", "question_only")
        result["oracle_evidence_accuracy_gain_over_question_only_↑"] = _mean(
            float(indexed["oracle_evidence"][record_id]["correct"])
            - float(indexed["question_only"][record_id]["correct"])
            for record_id in ids
        )
        result["oracle_evidence_gold_log_score_gain_over_question_only_↑"] = _mean(
            float(indexed["oracle_evidence"][record_id]["gold_log_score"])
            - float(indexed["question_only"][record_id]["gold_log_score"])
            for record_id in ids
        )

    if "oracle_evidence" in indexed and "shuffled_oracle_evidence" in indexed:
        ids = present("oracle_evidence", "shuffled_oracle_evidence")
        result["oracle_evidence_accuracy_gain_over_shuffled_oracle_evidence_↑"] = _mean(
            float(indexed["oracle_evidence"][record_id]["correct"])
            - float(indexed["shuffled_oracle_evidence"][record_id]["correct"])
            for record_id in ids
        )
        result[
            "oracle_evidence_gold_log_score_gain_over_shuffled_oracle_evidence_↑"
        ] = _mean(
            float(indexed["oracle_evidence"][record_id]["gold_log_score"])
            - float(indexed["shuffled_oracle_evidence"][record_id]["gold_log_score"])
            for record_id in ids
        )

    if "oracle_evidence" in indexed and "oracle_residual" in indexed:
        ids = present("oracle_evidence", "oracle_residual")
        result["oracle_evidence_accuracy_gain_over_oracle_residual_↑"] = _mean(
            float(indexed["oracle_evidence"][record_id]["correct"])
            - float(indexed["oracle_residual"][record_id]["correct"])
            for record_id in ids
        )
        result["oracle_evidence_gold_log_score_gain_over_oracle_residual_↑"] = _mean(
            float(indexed["oracle_evidence"][record_id]["gold_log_score"])
            - float(indexed["oracle_residual"][record_id]["gold_log_score"])
            for record_id in ids
        )

    if "mixture" in indexed and "predicted_evidence" in indexed:
        ids = present("mixture", "predicted_evidence")
        mixture_correct = [
            record_id for record_id in ids if indexed["mixture"][record_id]["correct"]
        ]
        result["conditional_sufficiency_given_mixture_correct_↑"] = accuracy(
            "predicted_evidence", mixture_correct
        )
        result["predicted_evidence_accuracy_gain_over_mixture_↑"] = _mean(
            float(indexed["predicted_evidence"][record_id]["correct"])
            - float(indexed["mixture"][record_id]["correct"])
            for record_id in ids
        )
        result["predicted_evidence_gold_log_score_gain_over_mixture_↑"] = _mean(
            float(indexed["predicted_evidence"][record_id]["gold_log_score"])
            - float(indexed["mixture"][record_id]["gold_log_score"])
            for record_id in ids
        )
        if "question_only" in indexed:
            causal_ids = [
                record_id
                for record_id in audio_dependent_ids
                if record_id in indexed["predicted_evidence"]
            ]
            result[
                "conditional_sufficiency_given_mixture_correct_question_only_wrong_↑"
            ] = accuracy("predicted_evidence", causal_ids)

    if "predicted_evidence" in indexed and "shuffled_evidence" in indexed:
        ids = present("predicted_evidence", "shuffled_evidence")
        result["predicted_evidence_accuracy_gain_over_shuffled_evidence_↑"] = _mean(
            float(indexed["predicted_evidence"][record_id]["correct"])
            - float(indexed["shuffled_evidence"][record_id]["correct"])
            for record_id in ids
        )
        result["predicted_evidence_gold_log_score_gain_over_shuffled_evidence_↑"] = (
            _mean(
                float(indexed["predicted_evidence"][record_id]["gold_log_score"])
                - float(indexed["shuffled_evidence"][record_id]["gold_log_score"])
                for record_id in ids
            )
        )

    if "mixture" in indexed and "predicted_residual" in indexed:
        ids = present("mixture", "predicted_residual")
        mixture_correct = [
            record_id for record_id in ids if indexed["mixture"][record_id]["correct"]
        ]
        leakage = accuracy("predicted_residual", mixture_correct)
        result["conditional_residual_leakage_given_mixture_correct_↓"] = leakage
        result["predicted_necessity_success_given_mixture_correct_↑"] = (
            None if leakage is None else 1.0 - leakage
        )
        result["predicted_necessity_accuracy_drop_↑"] = _mean(
            float(indexed["mixture"][record_id]["correct"])
            - float(indexed["predicted_residual"][record_id]["correct"])
            for record_id in ids
        )
        result["predicted_necessity_gold_log_score_drop_↑"] = _mean(
            float(indexed["mixture"][record_id]["gold_log_score"])
            - float(indexed["predicted_residual"][record_id]["gold_log_score"])
            for record_id in ids
        )
        if "question_only" in indexed:
            causal_ids = [
                record_id
                for record_id in audio_dependent_ids
                if record_id in indexed["predicted_residual"]
            ]
            causal_leakage = accuracy("predicted_residual", causal_ids)
            result[
                "conditional_residual_leakage_given_mixture_correct_question_only_wrong_↓"
            ] = causal_leakage
            result[
                "conditional_necessity_success_given_mixture_correct_question_only_wrong_↑"
            ] = (None if causal_leakage is None else 1.0 - causal_leakage)

    if (
        "mixture" in indexed
        and "question_only" in indexed
        and "oracle_evidence" in indexed
    ):
        causal_ids = [
            record_id
            for record_id in audio_dependent_ids
            if record_id in indexed["oracle_evidence"]
        ]
        result[
            "oracle_conditional_sufficiency_given_mixture_correct_question_only_wrong_↑"
        ] = accuracy("oracle_evidence", causal_ids)

    if (
        "mixture" in indexed
        and "question_only" in indexed
        and "oracle_residual" in indexed
    ):
        causal_ids = [
            record_id
            for record_id in audio_dependent_ids
            if record_id in indexed["oracle_residual"]
        ]
        causal_oracle_leakage = accuracy("oracle_residual", causal_ids)
        result[
            "oracle_conditional_residual_leakage_given_mixture_correct_question_only_wrong_↓"
        ] = causal_oracle_leakage
        result[
            "oracle_conditional_necessity_success_given_mixture_correct_question_only_wrong_↑"
        ] = (None if causal_oracle_leakage is None else 1.0 - causal_oracle_leakage)

    if "mixture" in indexed and "oracle_residual" in indexed:
        ids = present("mixture", "oracle_residual")
        mixture_correct = [
            record_id for record_id in ids if indexed["mixture"][record_id]["correct"]
        ]
        oracle_leakage = accuracy("oracle_residual", mixture_correct)
        result["oracle_necessity_success_given_mixture_correct_↑"] = (
            None if oracle_leakage is None else 1.0 - oracle_leakage
        )

    if "predicted_evidence" in indexed and "oracle_evidence" in indexed:
        ids = present("predicted_evidence", "oracle_evidence")
        predicted_accuracy = accuracy("predicted_evidence", ids)
        oracle_accuracy = accuracy("oracle_evidence", ids)
        assert predicted_accuracy is not None and oracle_accuracy is not None
        result["absolute_oracle_predicted_evidence_accuracy_gap_↓"] = abs(
            oracle_accuracy - predicted_accuracy
        )
        result["positive_oracle_evidence_headroom_accuracy_↓"] = max(
            0.0, oracle_accuracy - predicted_accuracy
        )

    control_names = {
        "silence": "silence_control_answerable_accuracy_↓",
        "question_only": "question_only_control_answerable_accuracy_↓",
        "shuffled_evidence": "shuffled_evidence_control_answerable_accuracy_↓",
        "shuffled_oracle_evidence": (
            "shuffled_oracle_evidence_control_answerable_accuracy_↓"
        ),
    }
    for condition, metric in control_names.items():
        if condition in indexed:
            ids = present(condition)
            result[metric] = accuracy(condition, ids)
    return result


def paired_subset_counts(items: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Expose denominators for conditional metrics without treating counts as scores."""

    indexed = _index_items(items)
    answerable_ids = sorted(
        {str(item["id"]) for item in items if not item["no_evidence"]}
    )
    result = {"answerable_records": len(answerable_ids)}
    if "mixture" in indexed:
        mixture_ids = [
            record_id for record_id in answerable_ids if record_id in indexed["mixture"]
        ]
        mixture_correct = [
            record_id
            for record_id in mixture_ids
            if indexed["mixture"][record_id]["correct"]
        ]
        result["mixture_correct_answerable_records"] = len(mixture_correct)
    if "mixture" in indexed and "question_only" in indexed:
        paired_ids = [
            record_id
            for record_id in answerable_ids
            if record_id in indexed["mixture"] and record_id in indexed["question_only"]
        ]
        result["mixture_question_only_paired_answerable_records"] = len(paired_ids)
        result["audio_dependent_mixture_correct_question_only_wrong_records"] = sum(
            bool(indexed["mixture"][record_id]["correct"])
            and not bool(indexed["question_only"][record_id]["correct"])
            for record_id in paired_ids
        )
    return result


def v5_paired_subgroup_metrics(
    items: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Summarize actual QA interventions over predeclared QCES-v5 slices."""

    if not items or not all("scene_family_id" in item for item in items):
        return {}

    dimensions: Dict[str, Sequence[Any]] = {
        "relation": ("after", "before", "first"),
        "variant_id": ("base", "order_swap", "anchor_drop"),
        "same_role_label": (False, True),
        "same_label_repeat": (False, True),
        "semantic_overlap": (False, True),
        "primary_counterfactual_probe": (False, True),
    }
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for field, values in dimensions.items():
        groups: Dict[str, Dict[str, Any]] = {}
        for value in values:
            selected = [item for item in items if item.get(field) == value]
            if not selected:
                continue
            record_ids = {str(item["id"]) for item in selected}
            answerable_ids = {
                str(item["id"]) for item in selected if not item["no_evidence"]
            }
            groups[str(value).lower()] = {
                "record_count": len(record_ids),
                "answerable_record_count": len(answerable_ids),
                "paired_metrics": paired_metrics(selected),
            }
        if groups:
            result[field] = groups
    return result


def bootstrap_paired_metric_summary(
    items: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    by_scene: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        cluster_id = item.get("scene_family_id", item["scene_id"])
        by_scene[str(cluster_id)].append(item)
    scene_ids = sorted(by_scene)
    if not scene_ids:
        return {}, {
            "requested_draws": samples,
            "independent_scene_clusters": 0,
            "cluster_unit": "scene_family_id_if_available_else_scene_id",
            "valid_draws_by_metric": {},
            "valid_fraction_by_metric": {},
        }
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = defaultdict(list)
    observed_metrics = set(paired_metrics(items))
    for _ in range(samples):
        selected: List[Mapping[str, Any]] = []
        for draw_index in range(len(scene_ids)):
            scene_id = rng.choice(scene_ids)
            # Prefixing IDs keeps duplicate bootstrap scenes as independent clusters.
            for item in by_scene[scene_id]:
                copy = dict(item)
                copy["id"] = f"draw{draw_index}:{item['id']}"
                selected.append(copy)
        for name, value in paired_metrics(selected).items():
            if value is not None:
                draws[name].append(float(value))
    result: Dict[str, List[float]] = {}
    metric_names = sorted(observed_metrics | set(draws))
    for name in metric_names:
        values = draws.get(name, [])
        if values:
            result[name] = [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
    coverage = {
        "requested_draws": samples,
        "independent_scene_clusters": len(scene_ids),
        "cluster_unit": "scene_family_id_if_available_else_scene_id",
        "valid_draws_by_metric": {
            name: len(draws.get(name, [])) for name in metric_names
        },
        "valid_fraction_by_metric": {
            name: len(draws.get(name, [])) / samples for name in metric_names
        },
    }
    return result, coverage


def bootstrap_paired_metrics(
    items: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> Dict[str, List[float]]:
    """Backward-compatible interval-only wrapper."""

    intervals, _ = bootstrap_paired_metric_summary(items, samples, seed)
    return intervals


def bootstrap_option_order_metric_summary(
    items: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Any]]:
    """Scene-family clustered intervals for isolated option-order controls."""

    observed, _ = option_order_control_summary(items)
    if not observed:
        return {}, {
            "requested_draws": samples,
            "independent_scene_clusters": 0,
            "cluster_unit": "scene_family_id_if_available_else_scene_id",
            "valid_draws_by_condition_and_metric": {},
            "valid_fraction_by_condition_and_metric": {},
        }
    by_scene: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        cluster_id = item.get("scene_family_id", item["scene_id"])
        by_scene[str(cluster_id)].append(item)
    scene_ids = sorted(by_scene)
    rng = random.Random(seed)
    draws: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for _ in range(samples):
        selected: List[Mapping[str, Any]] = []
        for draw_index in range(len(scene_ids)):
            scene_id = rng.choice(scene_ids)
            for item in by_scene[scene_id]:
                copy = dict(item)
                copy["id"] = f"draw{draw_index}:{item['id']}"
                selected.append(copy)
        draw_summaries, _ = option_order_control_summary(selected)
        for condition, metrics in draw_summaries.items():
            for name, value in metrics.items():
                if value is not None:
                    draws[condition][name].append(float(value))

    intervals: Dict[str, Dict[str, List[float]]] = {}
    valid_draws: Dict[str, Dict[str, int]] = {}
    valid_fractions: Dict[str, Dict[str, float]] = {}
    for condition, metrics in observed.items():
        intervals[condition] = {}
        valid_draws[condition] = {}
        valid_fractions[condition] = {}
        for name in sorted(metrics):
            values = draws[condition].get(name, [])
            valid_draws[condition][name] = len(values)
            valid_fractions[condition][name] = len(values) / samples
            if values:
                intervals[condition][name] = [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ]
    return intervals, {
        "requested_draws": samples,
        "independent_scene_clusters": len(scene_ids),
        "cluster_unit": "scene_family_id_if_available_else_scene_id",
        "valid_draws_by_condition_and_metric": valid_draws,
        "valid_fraction_by_condition_and_metric": valid_fractions,
    }


def validate_metric_metadata(
    condition_summaries: Mapping[str, Mapping[str, Any]],
    paired_summary: Mapping[str, Any],
    option_order_summaries: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    families = [
        (
            {key for summary in condition_summaries.values() for key in summary},
            CONDITION_METRIC_DEFINITIONS,
        ),
        (set(paired_summary), PAIRED_METRIC_DEFINITIONS),
    ]
    if option_order_summaries is not None:
        families.append(
            (
                {key for summary in option_order_summaries.values() for key in summary},
                OPTION_ORDER_METRIC_DEFINITIONS,
            )
        )
    for metrics, definitions in families:
        missing = metrics - set(definitions)
        if missing:
            raise ValueError(f"metric definitions are missing for: {sorted(missing)}")
        for name in metrics:
            if not name.endswith(("_↑", "_↓")):
                raise ValueError(f"metric lacks optimization arrow: {name}")
            direction = name[-1]
            if not definitions[name].startswith(direction):
                raise ValueError(f"metric description lacks matching direction: {name}")


def load_completed_items(
    path: Path, run_fingerprint: str
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    completed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    byte_offset = 0
    truncate_at: Optional[int] = None
    for index, encoded in enumerate(lines):
        line_start = byte_offset
        byte_offset += len(encoded)
        if not encoded.strip():
            continue
        try:
            item = json.loads(encoded)
        except (json.JSONDecodeError, UnicodeDecodeError):
            is_truncated_tail = index == len(lines) - 1 and not encoded.endswith(
                (b"\n", b"\r")
            )
            if is_truncated_tail:
                truncate_at = line_start
                break
            raise ValueError(f"invalid resumable JSONL at {path}:{index + 1}")
        if not isinstance(item, dict):
            raise ValueError(
                f"resumable JSONL item is not an object at {path}:{index + 1}"
            )
        if item.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"run fingerprint mismatch in {path}:{index + 1}")
        key = (str(item.get("id")), str(item.get("condition")))
        if key in completed:
            raise ValueError(f"duplicate completed item in {path}: {key}")
        completed[key] = item
    if truncate_at is not None:
        # A process can die between writes despite flush/fsync.  Merely ignoring
        # the partial line is insufficient: the next append would concatenate a
        # valid JSON object onto the broken prefix.  Repair the tail first.
        with path.open("r+b") as handle:
            handle.truncate(truncate_at)
            handle.flush()
            os.fsync(handle.fileno())
    elif raw and not raw.endswith((b"\n", b"\r")):
        # A complete final object can still be missing only its newline after a
        # crash.  Normalize it so any pending item starts on a fresh line.
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return completed


def append_item(path: Path, item: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_report(
    *,
    records: Sequence[AudioQARecord],
    conditions: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    run_fingerprint: str,
    metadata: Mapping[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["condition"])].append(item)
    summaries = {
        condition: condition_metrics(grouped[condition]) for condition in conditions
    }
    paired = paired_metrics(items)
    option_order_summaries, option_order_coverage = option_order_control_summary(items)
    validate_metric_metadata(summaries, paired, option_order_summaries)
    bootstrap_intervals, bootstrap_coverage = bootstrap_paired_metric_summary(
        items, bootstrap_samples, seed
    )
    (
        option_order_bootstrap_intervals,
        option_order_bootstrap_coverage,
    ) = bootstrap_option_order_metric_summary(items, bootstrap_samples, seed + 1)
    definitions = dict(CONDITION_METRIC_DEFINITIONS)
    definitions.update(PAIRED_METRIC_DEFINITIONS)
    definitions.update(OPTION_ORDER_METRIC_DEFINITIONS)
    used_definitions = {
        name: description
        for name, description in definitions.items()
        if any(name in summary for summary in summaries.values())
        or name in paired
        or any(name in summary for summary in option_order_summaries.values())
    }
    prediction_provenance = metadata.get("predictions")
    proxy_metrics: Dict[str, Any] = {
        "computed_by_this_audioqa_evaluator": False,
        "metrics": {},
        "interpretation": (
            "Waveform/temporal scores are model-free acoustic proxies, not "
            "evidence sufficiency or necessity measured by a QA model."
        ),
    }
    if isinstance(prediction_provenance, Mapping):
        reported_proxies = prediction_provenance.get("model_free_proxy_metrics")
        if isinstance(reported_proxies, Mapping):
            proxy_metrics["metrics"] = dict(reported_proxies)
            proxy_metrics["source"] = "separator evaluation_report.json"

    report = {
        "format": FORMAT_VERSION,
        "run_fingerprint": run_fingerprint,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "metric_definitions": used_definitions,
        "metric_families": {
            "actual_frozen_qa_model_metrics": {
                "json_paths": [
                    "condition_metrics",
                    "paired_metrics",
                    "v5_paired_subgroup_metrics",
                    "option_order_control_metrics_by_condition",
                ],
                "gold_answer_or_gold_option_index_used_as_model_input": False,
                "model_inputs": [
                    "question_text",
                    "five_unmarked_answer_options",
                    "condition_audio_or_question_only_control",
                ],
            },
            "model_free_acoustic_proxy_metrics": proxy_metrics,
        },
        "provenance": metadata,
        "counts": {
            "records": len(records),
            "answerable": sum(not record.no_evidence for record in records),
            "no_evidence": sum(record.no_evidence for record in records),
            "conditions": len(conditions),
            "completed_record_conditions": len(items),
        },
        "condition_metrics": summaries,
        "option_order_control_metrics_by_condition": option_order_summaries,
        "option_order_control_coverage_by_condition": option_order_coverage,
        "option_order_scene_bootstrap_95ci": option_order_bootstrap_intervals,
        "option_order_scene_bootstrap_coverage": option_order_bootstrap_coverage,
        "surface_control_pair_coverage_by_condition": {
            condition: _surface_control_pair_summary(grouped[condition])[1]
            for condition in conditions
        },
        "paired_metrics": paired,
        "paired_subset_counts": paired_subset_counts(items),
        "paired_scene_bootstrap_95ci": bootstrap_intervals,
        "paired_scene_bootstrap_coverage": bootstrap_coverage,
        "paired_independent_cluster_bootstrap_95ci": bootstrap_intervals,
        "paired_independent_cluster_bootstrap_coverage": bootstrap_coverage,
        "items_jsonl": "items.jsonl",
    }
    if all(isinstance(record, QCESV5Record) for record in records):
        report["v5_paired_subgroup_metrics"] = v5_paired_subgroup_metrics(items)
    return report


def _prediction_provenance(
    root: Optional[Path], report_override: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    if root is None:
        return None
    resolved = root.resolve()
    report = (
        report_override.resolve()
        if report_override is not None
        else resolved / "evaluation_report.json"
    )
    result: Dict[str, Any] = {"root": str(resolved)}
    if report.is_file():
        result["evaluation_report"] = str(report)
        result["evaluation_report_sha256"] = _sha256_file(report)
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            result["separator_checkpoint"] = payload.get("checkpoint")
        except (json.JSONDecodeError, UnicodeDecodeError):
            result["separator_checkpoint"] = None
    return result


def _validate_current_file_identity(identity: Any, description: str) -> Dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{description} identity is missing")
    path_value = identity.get("path")
    sha256 = identity.get("sha256")
    size_bytes = identity.get("size_bytes")
    if (
        not isinstance(path_value, str)
        or not isinstance(sha256, str)
        or not isinstance(size_bytes, int)
    ):
        raise ValueError(f"{description} identity is incomplete")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} file is missing: {path}")
    actual_size = path.stat().st_size
    actual_sha256 = _sha256_file(path)
    if actual_size != size_bytes or actual_sha256 != sha256:
        raise ValueError(f"{description} identity does not match current bytes")
    return {"path": str(path), "sha256": actual_sha256, "size_bytes": actual_size}


def _validate_foundation_cache_provenance(
    cache_identity: Any, manifest_identity: Mapping[str, Any]
) -> Dict[str, Any]:
    if not isinstance(cache_identity, Mapping):
        raise ValueError("foundation cache identity is missing")
    if cache_identity.get("contains_oracle_or_label_inputs") is not False:
        raise ValueError("foundation cache must explicitly exclude oracle/label inputs")
    binding = cache_identity.get("manifest_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("foundation cache manifest binding is missing")
    expected_sha = manifest_identity["sha256"]
    expected_size = manifest_identity["size_bytes"]
    for name in ("cache_declared", "run_expected"):
        value = binding.get(name)
        if (
            not isinstance(value, Mapping)
            or value.get("sha256") != expected_sha
            or value.get("size_bytes") != expected_size
        ):
            raise ValueError(f"foundation cache {name} does not match AudioQA manifest")
    checkpoint_binding = cache_identity.get("audiosep_checkpoint_binding")
    if not isinstance(checkpoint_binding, Mapping):
        raise ValueError("foundation cache AudioSep checkpoint binding is missing")
    declared = checkpoint_binding.get("cache_declared")
    expected = checkpoint_binding.get("run_expected")
    if not isinstance(declared, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("foundation cache AudioSep checkpoint identity is invalid")
    if declared.get("sha256") != expected.get("sha256") or declared.get(
        "size_bytes"
    ) != expected.get("size_bytes"):
        raise ValueError("foundation cache AudioSep checkpoint binding disagrees")

    verified_artifacts = {
        name: _validate_current_file_identity(cache_identity.get(name), name)
        for name in (
            "receipt",
            "question_feature_artifact",
            "scene_feature_artifact",
        )
    }
    return {
        "present": True,
        "format": cache_identity.get("format"),
        "directory": cache_identity.get("directory"),
        "manifest_sha256": expected_sha,
        "audiosep_checkpoint_sha256": expected.get("sha256"),
        "contains_oracle_or_label_inputs": False,
        "verified_artifacts": verified_artifacts,
    }


def strict_v5_prediction_provenance(
    root: Path,
    manifest: Path,
    records: Sequence[AudioQARecord],
    report_override: Optional[Path] = None,
    prediction_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind v5 predicted WAVs to an exact supported generator report.

    ``root`` always points to the directory whose immediate children are scene
    folders.  For multi-mode baselines, ``report_override`` may point to the
    parent report while ``prediction_mode`` selects the matching item rows.
    """

    if not records or not all(isinstance(record, QCESV5Record) for record in records):
        raise ValueError(
            "strict v5 prediction provenance requires only QCES v5 records"
        )
    resolved = root.resolve()
    report_path = (
        report_override.resolve()
        if report_override is not None
        else resolved / "evaluation_report.json"
    )
    if not report_path.is_file():
        raise FileNotFoundError(f"QCES v5 predicted conditions require {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid separator evaluation report") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("separator evaluation report must be an object")
    report_format = payload.get("format")
    supported_formats = {
        "qces_question_swap_eval_v1",
        "qces_v5_caption_planner_audiosep_eval_v1",
        "qces_v5_sam_audio_baselines_v2",
        "qces_v5_temporal_only_baselines_v1",
        "qces_v5_audiosep_baselines_v1",
        "qces_v6_pipeline_evaluation_v1",
    }
    if report_format not in supported_formats:
        raise ValueError("unsupported separator evaluation report format")
    schema_versions = {record.schema_version for record in records}
    declared_schema = payload.get("schema_version")
    declared_schemas = payload.get("schema_versions")
    if declared_schema is not None:
        schema_match = declared_schema in schema_versions
    elif isinstance(declared_schemas, list):
        schema_match = bool(schema_versions) and schema_versions.issubset(
            set(declared_schemas)
        )
    else:
        schema_match = False
    if len(schema_versions) != 1 or not schema_match:
        raise ValueError("separator evaluation report schema does not match manifest")
    report_manifest = payload.get("manifest")
    if (
        not isinstance(report_manifest, str)
        or Path(report_manifest).resolve() != manifest
    ):
        raise ValueError("separator evaluation report manifest path does not match")
    manifest_identity = {
        "path": str(manifest),
        "sha256": _sha256_file(manifest),
        "size_bytes": manifest.stat().st_size,
    }
    report_manifest_sha = payload.get("manifest_sha256")
    if (
        report_manifest_sha is not None
        and report_manifest_sha != manifest_identity["sha256"]
    ):
        raise ValueError("separator evaluation report manifest hash does not match")

    report_items = payload.get("items")
    if not isinstance(report_items, list):
        raise ValueError("separator evaluation report items are missing")
    selected_report_items: List[Mapping[str, Any]] = []
    for item in report_items:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ValueError("separator evaluation report has an invalid item")
        if prediction_mode is not None and item.get("mode") != prediction_mode:
            continue
        selected_report_items.append(item)
    if prediction_mode is not None and not selected_report_items:
        raise ValueError(
            f"separator evaluation report has no items for mode {prediction_mode!r}"
        )

    indexed_items: Dict[str, Mapping[str, Any]] = {}
    for item in selected_report_items:
        item_id = str(item["id"])
        if item_id in indexed_items:
            hint = "; select --predictions-mode" if prediction_mode is None else ""
            raise ValueError(
                "separator evaluation report has duplicate item IDs" + hint
            )
        indexed_items[item_id] = item
    expected_records = {record.sample_id: record for record in records}
    missing = set(expected_records) - set(indexed_items)
    if missing:
        raise ValueError(
            "separator evaluation report lacks selected IDs: "
            + ", ".join(sorted(missing)[:3])
        )
    for sample_id, record in expected_records.items():
        item = indexed_items[sample_id]
        if item.get("scene_id") != record.scene_id:
            raise ValueError("separator evaluation item scene ID mismatch")
        if item.get("scene_family_id") != record.scene_family_id:
            raise ValueError("separator evaluation item family ID mismatch")

    checkpoint_value: Any = None
    if report_format == "qces_question_swap_eval_v1":
        checkpoint_value = payload.get("checkpoint")
    elif report_format in {
        "qces_v5_caption_planner_audiosep_eval_v1",
        "qces_v5_audiosep_baselines_v1",
    }:
        checkpoint_value = payload.get("audiosep_checkpoint")
    elif report_format == "qces_v6_pipeline_evaluation_v1":
        # QCES-v6 freezes the separator and trains only the proposal head, so
        # the head checkpoint is the prediction-generating identity.  Oracle
        # modes have no head, and their identity is the frozen separator hash
        # already recorded in the report.
        checkpoint_value = payload.get("proposal_head") or payload.get(
            "audiosep_checkpoint"
        )
    elif report_format == "qces_v5_sam_audio_baselines_v2":
        checkpoint_value = payload.get("model_checkpoint")
    elif report_format == "qces_v5_temporal_only_baselines_v1":
        predicted_provenance = payload.get("predicted_span_provenance")
        if isinstance(predicted_provenance, Mapping):
            checkpoint_value = predicted_provenance.get("checkpoint")

    checkpoint_identity: Optional[Dict[str, Any]] = None
    if isinstance(checkpoint_value, Mapping):
        checkpoint_identity = _validate_current_file_identity(
            checkpoint_value, "prediction-generator checkpoint"
        )
    elif isinstance(checkpoint_value, str):
        checkpoint_path = Path(checkpoint_value).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "prediction-generator checkpoint referenced by report is missing: "
                f"{checkpoint_path}"
            )
        checkpoint_identity = {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        }
    elif report_format != "qces_v5_temporal_only_baselines_v1":
        raise ValueError("prediction-generator checkpoint is missing")
    elif prediction_mode is not None and prediction_mode.startswith(
        "question_predicted_"
    ):
        raise ValueError("predicted temporal mode lacks its QCES checkpoint identity")

    if report_format == "qces_question_swap_eval_v1":
        summary = payload.get("summary_with_directions")
    elif report_format == "qces_v5_caption_planner_audiosep_eval_v1":
        summary = payload.get("summary")
    elif report_format == "qces_v6_pipeline_evaluation_v1":
        summaries = payload.get("summaries_by_mode")
        summary = (
            summaries.get(prediction_mode)
            if isinstance(summaries, Mapping) and prediction_mode is not None
            else None
        )
    else:
        summaries = payload.get("summaries")
        summary = (
            summaries.get(prediction_mode)
            if isinstance(summaries, Mapping) and prediction_mode is not None
            else None
        )
    if not isinstance(summary, Mapping):
        raise ValueError("separator report lacks directed proxy metrics")
    proxy_metrics: Dict[str, float] = {}
    for name, value in summary.items():
        if not isinstance(name, str) or not name.endswith(("_↑", "_↓")):
            continue
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("separator proxy metric must be numeric")
        proxy_metrics[name] = float(value)
    if not proxy_metrics:
        raise ValueError("separator report has no directed numeric proxy metrics")

    foundation = payload.get("foundation_features")
    cache_provenance: Dict[str, Any] = {"present": False}
    if foundation is not None:
        if not isinstance(foundation, Mapping):
            raise ValueError("separator foundation feature provenance is invalid")
        cache_provenance = _validate_foundation_cache_provenance(
            foundation.get("cache_identity"), manifest_identity
        )

    return {
        "root": str(resolved),
        "evaluation_report": str(report_path),
        "evaluation_report_sha256": _sha256_file(report_path),
        "evaluation_report_format": report_format,
        "prediction_mode": prediction_mode,
        "manifest": manifest_identity,
        "separator_checkpoint": checkpoint_identity,
        "foundation_feature_cache": cache_provenance,
        "selected_item_count": len(expected_records),
        "selected_item_ids_sha256": _sha256_json(sorted(expected_records)),
        "model_free_proxy_metrics": proxy_metrics,
        "validation_status": "strict_v5_binding_passed",
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = args.manifest.resolve()
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else manifest.parent
    )
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root does not exist: {dataset_root}")
    records = load_records(manifest, args.split, args.max_records)
    is_v5 = all(isinstance(record, QCESV5Record) for record in records)
    if is_v5 and Path(args.model).expanduser().exists() and not args.hash_model_weights:
        raise SystemExit(
            "QCES v5 requires --hash-model-weights for a local QA checkpoint; "
            "a remote model already requires an explicit --revision"
        )
    predictions_root = (
        args.predictions_root.resolve() if args.predictions_root else None
    )
    predictions_report = (
        args.predictions_report.resolve() if args.predictions_report else None
    )
    uses_predictions = bool(
        {_base_audio_condition(value) for value in args.conditions}
        & PREDICTED_CONDITIONS
    )
    if is_v5 and uses_predictions:
        assert predictions_root is not None
        prediction_provenance = strict_v5_prediction_provenance(
            predictions_root,
            manifest,
            records,
            report_override=predictions_report,
            prediction_mode=args.predictions_mode,
        )
    else:
        prediction_provenance = _prediction_provenance(
            predictions_root, predictions_report
        )
    descriptors = build_input_descriptors(
        records, dataset_root, predictions_root, args.conditions
    )
    source_inventory = [asdict(descriptors[key]) for key in sorted(descriptors)]
    script_path = Path(__file__).resolve()
    script_sha256 = _sha256_file(script_path)
    model_source = _model_source_provenance(args.model, args.hash_model_weights)
    run_config = {
        "format": FORMAT_VERSION,
        "auditor": args.auditor,
        "prompt_version": PROMPT_VERSION,
        "scoring_version": SCORING_VERSION,
        "evaluator_script_sha256": script_sha256,
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "dataset_root": str(dataset_root),
        "requested_split": args.split,
        "max_records": args.max_records,
        "selected_record_ids_sha256": _sha256_json(
            [record.sample_id for record in records]
        ),
        "conditions": list(args.conditions),
        "audio_conditions": list(args.audio_conditions),
        "option_order_control_conditions": list(args.option_order_control_conditions),
        "option_order_control": {
            "enabled": bool(args.option_order_control_conditions),
            "permutation": "nonzero_cyclic_shift_sha256(record_id,seed)",
            "uses_gold_answer_or_position": False,
            "question_text_changed": False,
            "audio_changed": False,
        },
        "source_inventory_sha256": _sha256_json(source_inventory),
        "model": args.model,
        "revision": args.revision,
        "model_source_inventory_sha256": _sha256_json(model_source),
        "prediction_provenance_sha256": _sha256_json(prediction_provenance),
        "predictions_report": (
            str(predictions_report) if predictions_report is not None else None
        ),
        "predictions_mode": args.predictions_mode,
        "hash_model_weights": args.hash_model_weights,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "device": args.device,
        "device_map": args.device_map,
        "attention_implementation": args.attention_implementation,
        "local_files_only": args.local_files_only,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "runtime_package_versions": {
            name: _package_version(name)
            for name in ("transformers", "torch", "numpy", "scipy", "soundfile")
        },
    }
    run_fingerprint = _sha256_json(run_config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "run_fingerprint": run_fingerprint,
                    "records": len(records),
                    "record_conditions": len(descriptors),
                },
                indent=2,
            )
        )
        return

    output_dir = args.output_dir.resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    items_path = output_dir / "items.jsonl"
    report_path = output_dir / "evaluation_report.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_fingerprint") != run_fingerprint:
            raise SystemExit(
                f"existing output has another run fingerprint: {output_dir}; use --overwrite"
            )
    elif any(output_dir.iterdir()):
        raise SystemExit(
            f"output is non-empty but has no run_metadata.json: {output_dir}; use --overwrite"
        )
    else:
        metadata = {
            "run_fingerprint": run_fingerprint,
            "run_config": run_config,
            "source_inventory": source_inventory,
            "script": str(script_path),
            "script_sha256": script_sha256,
            "git": _git_provenance(),
            "model_source": model_source,
            "predictions": prediction_provenance,
            "model_input_contract": {
                "question_text": True,
                "five_unmarked_answer_options": True,
                "condition_audio_or_question_only_control": True,
                "gold_answer": False,
                "gold_option_index": False,
                "post_hoc_gold_scoring_only": True,
            },
            "runtime_model": None,
        }
        _atomic_json(metadata_path, metadata)

    completed = load_completed_items(items_path, run_fingerprint)
    expected = set(descriptors)
    unexpected = set(completed) - expected
    if unexpected:
        raise ValueError(
            f"items.jsonl contains unexpected record/condition keys: {sorted(unexpected)[:3]}"
        )
    pending = [key for key in sorted(expected) if key not in completed]
    scorer: Optional[OptionScorer] = None
    if pending:
        scorer_class = {
            "af3": AudioFlamingo3OptionScorer,
            "qwen2_audio": Qwen2AudioOptionScorer,
            "phi4mm": Phi4MMOptionScorer,
        }[args.auditor]
        scorer = scorer_class(
            model_name=args.model,
            revision=args.revision,
            quantization=args.quantization,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
            attention_implementation=args.attention_implementation,
            local_files_only=args.local_files_only,
            seed=args.seed,
        )
        metadata = dict(metadata)
        metadata["runtime_model"] = dict(scorer.provenance())
        _atomic_json(metadata_path, metadata)
        record_map = {record.sample_id: record for record in records}
        for progress, key in enumerate(pending, start=1):
            record_id, _ = key
            descriptor = descriptors[key]
            waveform, sample_rate = materialize_input(descriptor)
            options = presented_answer_options(
                record_map[record_id], descriptor.condition, args.seed
            )
            score = score_record_without_gold_inputs(
                scorer,
                record_map[record_id],
                waveform,
                sample_rate,
                options,
            )
            item = make_item(
                record_map[record_id],
                descriptor,
                score,
                run_fingerprint,
                options,
            )
            append_item(items_path, item)
            completed[key] = item
            print(
                f"[{progress}/{len(pending)}] {record_id} {descriptor.condition}: "
                f"{item['predicted_answer']} ({'correct' if item['correct'] else 'wrong'})",
                flush=True,
            )

    ordered_items = [completed[key] for key in sorted(expected)]
    report = build_report(
        records=records,
        conditions=args.conditions,
        items=ordered_items,
        run_fingerprint=run_fingerprint,
        metadata=metadata,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _atomic_json(report_path, report)
    print(json.dumps(report["paired_metrics"], indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
