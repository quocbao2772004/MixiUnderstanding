"""Frozen BEATs/AudioSet auditor for QCES evidence and complement stems.

BEATs is evaluation-only.  Gold event labels select output coordinates after
inference; they are never inputs to QCES or BEATs.  This gives an
architecturally distinct event-recognition audit beside the generative AF3 QA
auditor without reusing AudioSep's CLAP condition space.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torchaudio.functional as audio_functional


EXPECTED_SOURCE_COMMIT = "833df7e7832e5064a281131ee64a481afa8e5b95"
EXPECTED_CHECKPOINT_SHA256 = (
    "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c"
)
EXPECTED_CHECKPOINT_SIZE = 363_145_291
EXPECTED_LABELS_SHA256 = (
    "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429"
)
EXPECTED_CLASS_COUNT = 527
MIRROR_REPOSITORY = "WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2"
MIRROR_REVISION = "172e4071a2912392d9151c70750b9a12f55fc54e"
OFFICIAL_SOURCE_URL = "https://github.com/microsoft/unilm/tree/master/beats"
OFFICIAL_CHECKPOINT_PAGE = (
    "https://github.com/microsoft/unilm/blob/master/beats/README.md"
)
OFFICIAL_AUDIOSET_LABELS_URL = (
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "class_labels_indices.csv"
)

# FSD50K uses the more specific child name.  The released AS2M predictor has a
# single Cymbal output, so this one many-to-one ontology projection is explicit.
QCES_AUDIOSET_ALIASES = {"Crash_cymbal": "Cymbal"}

BEATS_METRIC_DIRECTIONS = {
    "oracle_required_probability_evidence": "↑",
    "oracle_required_probability_residual": "↓",
    "oracle_evidence_residual_contrast": "↑",
    "oracle_vs_mixture_sufficiency_delta": "↑",
    "mixture_vs_oracle_residual_necessity_delta": "↑",
    "oracle_excluded_probability_evidence": "↓",
    "oracle_excluded_suppression_delta": "↑",
    "predicted_required_probability_evidence": "↑",
    "predicted_required_probability_residual": "↓",
    "predicted_evidence_residual_contrast": "↑",
    "predicted_vs_mixture_sufficiency_delta": "↑",
    "mixture_vs_predicted_residual_necessity_delta": "↑",
    "predicted_excluded_probability_evidence": "↓",
    "predicted_excluded_suppression_delta": "↑",
}


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing BEATs asset: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def normalize_event_label(label: str) -> str:
    if not isinstance(label, str) or not label.strip():
        raise ValueError("event label must be a non-empty string")
    tokens = re.sub(r"[^a-z0-9]+", " ", label.lower()).split()
    # FSD50K class slugs spell comma-separated AudioSet synonyms with ``and``
    # (e.g. Chewing_and_mastication).  The 527-class AudioSet display names use
    # punctuation instead.  Removing only the standalone conjunction yields a
    # collision-free mapping across the pinned checkpoint vocabulary.
    return " ".join(token for token in tokens if token != "and")


def load_audioset_names(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"AudioSet label file has {len(rows)} rows, expected {EXPECTED_CLASS_COUNT}"
        )
    for row in rows:
        mid = row.get("mid")
        name = row.get("display_name")
        if not isinstance(mid, str) or not isinstance(name, str) or not mid or not name:
            raise ValueError("AudioSet label file contains an invalid row")
        if mid in result:
            raise ValueError(f"duplicate AudioSet MID: {mid}")
        result[mid] = name
    return result


def resolve_qces_label_indices(
    qces_labels: Iterable[str],
    checkpoint_label_dict: Mapping[int, str],
    audioset_names_by_mid: Mapping[str, str],
) -> Dict[str, int]:
    if len(checkpoint_label_dict) != EXPECTED_CLASS_COUNT:
        raise ValueError("BEATs checkpoint does not expose 527 output labels")
    by_normalized_name: Dict[str, list[int]] = {}
    for raw_index, mid in checkpoint_label_dict.items():
        index = int(raw_index)
        name = audioset_names_by_mid.get(str(mid))
        if name is None:
            raise ValueError(f"BEATs checkpoint MID is absent from AudioSet: {mid}")
        by_normalized_name.setdefault(normalize_event_label(name), []).append(index)

    resolved = {}
    for qces_label in sorted(set(qces_labels)):
        query = QCES_AUDIOSET_ALIASES.get(qces_label, qces_label)
        matches = by_normalized_name.get(normalize_event_label(query), [])
        if len(matches) != 1:
            raise ValueError(
                f"QCES label {qces_label!r} maps to {len(matches)} BEATs outputs"
            )
        resolved[qces_label] = matches[0]
    return resolved


def _git_head(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def validate_beats_assets(
    repository_root: Path, checkpoint_path: Path, labels_path: Path
) -> Dict[str, Any]:
    repository_root = repository_root.resolve()
    source_commit = _git_head(repository_root)
    if source_commit != EXPECTED_SOURCE_COMMIT:
        raise ValueError(
            f"BEATs source commit mismatch: {source_commit} != {EXPECTED_SOURCE_COMMIT}"
        )
    checkpoint = file_identity(checkpoint_path)
    labels = file_identity(labels_path)
    if (
        checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256
        or checkpoint["size_bytes"] != EXPECTED_CHECKPOINT_SIZE
    ):
        raise ValueError("BEATs checkpoint bytes do not match the pinned AS2M cpt2")
    if labels["sha256"] != EXPECTED_LABELS_SHA256:
        raise ValueError("AudioSet class-label bytes do not match the pin")
    source_files = {}
    for relative in (
        "beats/BEATs.py",
        "beats/backbone.py",
        "beats/modules.py",
        "LICENSE",
    ):
        source_files[relative] = file_identity(repository_root / relative)
    return {
        "source": {
            "repository": str(repository_root),
            "commit": source_commit,
            "official_url": OFFICIAL_SOURCE_URL,
            "license": "MIT",
            "files": source_files,
        },
        "checkpoint": {
            **checkpoint,
            "model": "BEATs iter3+ AS2M fine-tuned on AS2M cpt2",
            "official_index": OFFICIAL_CHECKPOINT_PAGE,
            "byte_mirror_repository": MIRROR_REPOSITORY,
            "byte_mirror_revision": MIRROR_REVISION,
        },
        "audioset_labels": {
            **labels,
            "official_url": OFFICIAL_AUDIOSET_LABELS_URL,
            "license": "CC-BY-4.0 / ontology CC-BY-SA-4.0",
        },
    }


@dataclass
class BEATsAudioSetAuditor:
    model: torch.nn.Module
    label_indices: Dict[str, int]
    checkpoint_label_dict: Dict[int, str]
    provenance: Dict[str, Any]
    device: torch.device

    @classmethod
    def from_assets(
        cls,
        repository_root: Path,
        checkpoint_path: Path,
        labels_path: Path,
        qces_labels: Iterable[str],
        device: torch.device,
    ) -> "BEATsAudioSetAuditor":
        provenance = validate_beats_assets(
            repository_root, checkpoint_path, labels_path
        )
        checkpoint = torch.load(
            checkpoint_path.resolve(), map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, Mapping) or not {
            "cfg",
            "model",
            "label_dict",
        }.issubset(checkpoint):
            raise ValueError("invalid BEATs checkpoint structure")
        raw_label_dict = checkpoint["label_dict"]
        if not isinstance(raw_label_dict, Mapping):
            raise ValueError("BEATs checkpoint lacks a label dictionary")
        label_dict = {int(index): str(mid) for index, mid in raw_label_dict.items()}
        labels_by_mid = load_audioset_names(labels_path.resolve())
        label_indices = resolve_qces_label_indices(
            qces_labels, label_dict, labels_by_mid
        )

        source_path = repository_root.resolve() / "beats"
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        beats_module = importlib.import_module("BEATs")
        config = beats_module.BEATsConfig(checkpoint["cfg"])
        model = beats_module.BEATs(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().to(device)
        if any(parameter.requires_grad for parameter in model.parameters()):
            model.requires_grad_(False)
        provenance["mapping"] = {
            "qces_label_count_↑": len(label_indices),
            "checkpoint_output_count_↑": len(label_dict),
            "explicit_aliases": dict(QCES_AUDIOSET_ALIASES),
            "normalization": (
                "lowercase alphanumerics; punctuation/underscores and standalone "
                "conjunction 'and' removed; exact unique match required"
            ),
            "gold_usage_boundary": (
                "gold labels select BEATs output coordinates after inference only; "
                "they are not QCES or BEATs inputs"
            ),
        }
        return cls(model, label_indices, label_dict, provenance, device)

    @torch.inference_mode()
    def score(
        self,
        waveforms: torch.Tensor,
        sample_rate: int,
        stream_batch_size: int,
    ) -> torch.Tensor:
        if waveforms.ndim != 2 or waveforms.size(0) <= 0:
            raise ValueError("BEATs waveforms must have shape [streams, samples]")
        if sample_rate <= 0 or stream_batch_size <= 0:
            raise ValueError("sample rate and stream batch size must be positive")
        if not bool(torch.isfinite(waveforms).all()):
            raise ValueError("BEATs input contains NaN or Inf")
        audio = waveforms.to(self.device, dtype=torch.float32)
        if sample_rate != 16_000:
            audio = audio_functional.resample(audio, sample_rate, 16_000)
        chunks = []
        for start in range(0, audio.size(0), stream_batch_size):
            probabilities, _ = self.model.extract_features(
                audio[start : start + stream_batch_size]
            )
            chunks.append(probabilities.float().cpu())
        result = torch.cat(chunks, dim=0)
        if result.shape != (waveforms.size(0), EXPECTED_CLASS_COUNT):
            raise RuntimeError(f"unexpected BEATs output shape: {tuple(result.shape)}")
        return result


def score_qces_streams(
    probabilities: torch.Tensor,
    required_labels: Sequence[str],
    excluded_labels: Sequence[str],
    label_indices: Mapping[str, int],
    residual_event_labels: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Score X, E*, R*, E, R probabilities in this exact row order."""

    if probabilities.shape != (5, EXPECTED_CLASS_COUNT):
        raise ValueError("stream probabilities must have shape [5, 527]")
    required = sorted(set(required_labels))
    if not required:
        raise ValueError("answerable BEATs scoring requires at least one label")
    missing = sorted((set(required) | set(excluded_labels)) - set(label_indices))
    if missing:
        raise ValueError(f"BEATs label mapping is incomplete: {missing}")
    required_indices = [label_indices[label] for label in required]

    def required_min(stream: int) -> float:
        return float(probabilities[stream, required_indices].min())

    x, oracle_e, oracle_r, predicted_e, predicted_r = (
        required_min(index) for index in range(5)
    )
    result = {
        "required_labels": required,
        "excluded_labels": sorted(set(excluded_labels) - set(required)),
        "required_probability_mixture": x,
        "oracle_required_probability_evidence": oracle_e,
        "oracle_required_probability_residual": oracle_r,
        "oracle_evidence_residual_contrast": oracle_e - oracle_r,
        "oracle_vs_mixture_sufficiency_delta": oracle_e - x,
        "mixture_vs_oracle_residual_necessity_delta": x - oracle_r,
        "predicted_required_probability_evidence": predicted_e,
        "predicted_required_probability_residual": predicted_r,
        "predicted_evidence_residual_contrast": predicted_e - predicted_r,
        "predicted_vs_mixture_sufficiency_delta": predicted_e - x,
        "mixture_vs_predicted_residual_necessity_delta": x - predicted_r,
    }
    excluded = result["excluded_labels"]
    if excluded:
        excluded_indices = [label_indices[label] for label in excluded]
        x_excluded = float(probabilities[0, excluded_indices].max())
        oracle_e_excluded = float(probabilities[1, excluded_indices].max())
        e_excluded = float(probabilities[3, excluded_indices].max())
        result.update(
            excluded_probability_mixture=x_excluded,
            oracle_excluded_probability_evidence=oracle_e_excluded,
            oracle_excluded_suppression_delta=x_excluded - oracle_e_excluded,
            predicted_excluded_probability_evidence=e_excluded,
            predicted_excluded_suppression_delta=x_excluded - e_excluded,
        )
    else:
        result.update(
            excluded_probability_mixture=None,
            oracle_excluded_probability_evidence=None,
            oracle_excluded_suppression_delta=None,
            predicted_excluded_probability_evidence=None,
            predicted_excluded_suppression_delta=None,
        )
    if residual_event_labels is not None:
        required_in_residual = sorted(set(required) & set(residual_event_labels))
        result.update(
            required_labels_present_in_oracle_residual=required_in_residual,
            class_separable_for_event_presence=not required_in_residual,
            class_scope_interpretation=(
                "BEATs class-presence necessity is interpretable"
                if not required_in_residual
                else (
                    "same required class has a legitimate non-evidence event in "
                    "the oracle residual; BEATs cannot distinguish event instances"
                )
            ),
        )
    return result


def summarize_beats_items(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("BEATs summary requires answerable items")
    summary: Dict[str, float] = {}
    medians: Dict[str, float] = {}
    for metric in BEATS_METRIC_DIRECTIONS:
        values = [
            float(item[metric])
            for item in items
            if isinstance(item.get(metric), (int, float))
            and math.isfinite(float(item[metric]))
        ]
        if not values:
            if "_excluded_" in metric:
                continue
            raise ValueError(f"BEATs items lack metric: {metric}")
        summary[metric] = sum(values) / len(values)
        medians[metric] = float(median(values))
    result = {
        "answerable_records_↑": len(items),
        "required_role_reduction": "minimum probability across unique role labels",
        "aggregate": "macro mean over answerable records",
        "summary": summary,
        "summary_with_directions": {
            f"{metric}_{BEATS_METRIC_DIRECTIONS[metric]}": value
            for metric, value in summary.items()
        },
        "median_diagnostics": medians,
        "claim_boundary": (
            "frozen AudioSet event-presence audit; not a second generative QA model "
            "and not proof of causal identification"
        ),
    }
    scoped = [
        item
        for item in items
        if isinstance(item.get("class_separable_for_event_presence"), bool)
    ]
    if scoped:
        separable = [
            item for item in scoped if item["class_separable_for_event_presence"]
        ]
        repeated = [
            item for item in scoped if not item["class_separable_for_event_presence"]
        ]

        def key_scope(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
            selected = (
                "oracle_evidence_residual_contrast",
                "mixture_vs_oracle_residual_necessity_delta",
                "predicted_evidence_residual_contrast",
                "mixture_vs_predicted_residual_necessity_delta",
            )
            directed: Dict[str, float] = {}
            for metric in selected:
                values = [
                    float(item[metric])
                    for item in rows
                    if isinstance(item.get(metric), (int, float))
                    and math.isfinite(float(item[metric]))
                ]
                if values:
                    directed[f"{metric}_{BEATS_METRIC_DIRECTIONS[metric]}"] = sum(
                        values
                    ) / len(values)
            return {"records": len(rows), "summary_with_directions": directed}

        result["class_presence_scope"] = {
            "primary_interpretable_subset": (
                "class_separable: no required event class remains in oracle residual"
            ),
            "class_separable": key_scope(separable),
            "same_required_class_present_in_residual": key_scope(repeated),
            "warning": (
                "Do not interpret BEATs residual probability as event-instance "
                "leakage when the oracle residual legitimately contains another "
                "occurrence of the required class."
            ),
        }
    return result
