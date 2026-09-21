#!/usr/bin/env python3
"""Train and evaluate Q-DOR on leakage-safe dense detector features.

This is a v2 sidecar pipeline.  It consumes only feature exports produced by
``export_qces_detector_dense_features.py`` and never imports or mutates the
historical proposal/ranker experiments.  Train and development scene lists
are mandatory and must be disjoint.

Training is deliberately factorized:

* stage A uses the oracle anchor and answer label for temporal teacher forcing;
* stage B linearly replaces those intermediates with model predictions.

The answer class and the answerability/NONE decision have separate losses.
Consequently an answerable row whose label is unavailable is a data error; it
is never silently converted into a NONE target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.benchmark_integrity import (
    BalancedQAItem,
    build_balanced_qa_items,
)
from mixi_understanding.qces.dense_event_qa import (
    FRAME_HOP_SECONDS,
    RELATION_AFTER,
    RELATION_BEFORE,
    DenseTemporalReasoner,
    DenseTemporalReasonerConfig,
    intervals_to_40ms_mask,
)


DENSE_INDEX_FORMAT = "qces_detector_dense_features_v2"
DENSE_SHARD_FORMAT = "qces_detector_dense_features_shard_v2"
SCHEMA_VERSION = 2
NUM_FRAMES = 250
FIXED_AUDIO_SECONDS = 10.0
RECEIPT_FORMAT = "qces_qdor_dense_training_receipt_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-index", type=Path, action="append", required=True)
    parser.add_argument("--train-scene-list", type=Path, required=True)
    parser.add_argument("--dev-scene-list", type=Path, required=True)
    parser.add_argument(
        "--train-qa-manifest",
        type=Path,
        help=(
            "Explicit conditionally-balanced QA JSONL from the final clean-scene "
            "builder. Must be supplied together with --dev-qa-manifest."
        ),
    )
    parser.add_argument(
        "--dev-qa-manifest",
        type=Path,
        help=(
            "Explicit conditionally-balanced development QA JSONL. When present, "
            "the historical all-adjacencies builder is not called."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--oracle-gate-receipt",
        type=Path,
        help=(
            "JSON from an oracle-span dense-feature label probe. Training is "
            "refused without a passing receipt unless the explicit debug override is used."
        ),
    )
    parser.add_argument("--min-oracle-answer-top1", type=float, default=0.70)
    parser.add_argument(
        "--allow-failed-oracle-gate",
        action="store_true",
        help="DEBUG ONLY: train despite a missing/failing oracle-span representation gate.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs-stage-a", type=int, default=4)
    parser.add_argument("--epochs-stage-b", type=int, default=12)
    parser.add_argument("--stage-b-teacher-start", type=float, default=0.9)
    parser.add_argument("--stage-b-teacher-end", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--logit-adjustment-tau", type=float, default=0.5)
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument("--prior-smoothing", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-answerable-per-scene", type=int, default=4)
    parser.add_argument("--max-no-evidence-per-scene", type=int, default=2)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload selected fp16 scenes into RAM (recommended for shuffled QA rows).",
    )
    parser.add_argument("--shard-cache-size", type=int, default=8)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_torch(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_scene_list(path: Path) -> list[str]:
    """Load an ordered, duplicate-free scene allowlist."""

    resolved = path.resolve()
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw = payload
        elif isinstance(payload, Mapping) and isinstance(payload.get("scene_ids"), list):
            raw = payload["scene_ids"]
        else:
            raise ValueError(f"scene JSON must be a list or contain scene_ids: {resolved}")
        values = [str(value).strip() for value in raw if str(value).strip()]
    elif resolved.suffix.lower() == ".jsonl":
        values = []
        for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            scene_id = str(row.get("scene_id") or "").strip()
            if not scene_id:
                raise ValueError(f"missing scene_id at {resolved}:{line_number}")
            values.append(scene_id)
    else:
        values = [
            line.strip()
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not values:
        raise ValueError(f"empty scene list: {resolved}")
    duplicates = [key for key, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate scene ids in {resolved}: {duplicates[:5]}")
    return values


def load_oracle_gate(
    path: Path | None,
    *,
    minimum_top1: float,
    allow_failed: bool,
) -> dict[str, Any]:
    """Enforce the representation ceiling before spending a full training run.

    The gate receipt must expose one of the deliberately narrow metric names
    below.  This avoids accidentally treating an unrelated ``accuracy`` field
    as oracle-span answer-label accuracy.
    """

    if not 0.0 <= minimum_top1 <= 1.0:
        raise ValueError("minimum oracle answer top1 must be in [0, 1]")
    if path is None:
        result = {
            "provided": False,
            "passed": False,
            "minimum_top1": minimum_top1,
            "reason": "missing oracle-span representation receipt",
        }
        if not allow_failed:
            raise RuntimeError(
                "Q-DOR training refused: provide --oracle-gate-receipt with "
                f"oracle answer top1 >= {minimum_top1:.3f}, or use "
                "--allow-failed-oracle-gate for an explicitly non-paper debug run"
            )
        result["debug_override"] = True
        return result
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    candidates: list[tuple[str, Any]] = []
    for key in (
        "oracle_span_answer_top1",
        "oracle_window_answer_top1",
        "oracle_answer_label_top1",
        "oracle_top1",
    ):
        if key in payload:
            candidates.append((key, payload[key]))
    for container_name in ("dev", "metrics", "best_dev", "oracle_probe"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            for key in (
                "oracle_span_answer_top1",
                "oracle_window_answer_top1",
                "oracle_answer_label_top1",
                "oracle_top1",
            ):
                if key in container:
                    candidates.append((f"{container_name}.{key}", container[key]))
    if len(candidates) != 1:
        raise ValueError(
            "oracle gate receipt must expose exactly one supported oracle top1 metric; "
            f"found {[name for name, _ in candidates]}"
        )
    metric_name, raw_value = candidates[0]
    top1 = float(raw_value)
    if not math.isfinite(top1) or not 0.0 <= top1 <= 1.0:
        raise ValueError(f"invalid oracle gate top1: {raw_value!r}")
    passed = top1 >= minimum_top1
    result = {
        "provided": True,
        "receipt": str(resolved),
        "receipt_sha256": _sha256_file(resolved),
        "metric": metric_name,
        "top1": top1,
        "minimum_top1": minimum_top1,
        "passed": passed,
    }
    if not passed and not allow_failed:
        raise RuntimeError(
            f"Q-DOR training refused: oracle-span answer top1={top1:.4f} is below "
            f"the required {minimum_top1:.4f}; rebuild the supported ontology/detector first"
        )
    result["debug_override"] = bool(not passed and allow_failed)
    return result


def _safe_shard_path(index_path: Path, relative: str) -> Path:
    if not relative:
        raise ValueError(f"empty shard path in {index_path}")
    root = index_path.parent.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"shard escapes dense index directory: {relative}")
    return path


class DenseFeatureStore:
    """Verified random access over one or more v2 dense-feature exports."""

    def __init__(self, index_paths: Sequence[Path], *, cache_size: int = 8) -> None:
        if not index_paths:
            raise ValueError("at least one dense index is required")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.cache_size = cache_size
        self.labels: list[str] | None = None
        self.feature_dim: int | None = None
        self._locations: dict[str, tuple[Path, int]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._cache: OrderedDict[Path, Mapping[str, Any]] = OrderedDict()
        self.index_paths = [path.resolve() for path in index_paths]
        self.index_sha256: dict[str, str] = {}
        for index_path in self.index_paths:
            self._scan_index(index_path)
        assert self.labels is not None and self.feature_dim is not None

    def _load_shard(self, path: Path) -> Mapping[str, Any]:
        cached = self._cache.pop(path, None)
        if cached is not None:
            self._cache[path] = cached
            return cached
        # weights_only rejects arbitrary globals and is mandatory for a feature
        # cache which may have been produced outside this training process.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"dense shard is not a mapping: {path}")
        if self.cache_size > 0:
            self._cache[path] = payload
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return payload

    def _scan_index(self, index_path: Path) -> None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("format") != DENSE_INDEX_FORMAT or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"not a dense-feature v2 index: {index_path}")
        grid = payload.get("grid") or {}
        if (
            int(grid.get("num_frames", -1)) != NUM_FRAMES
            or not math.isclose(float(grid.get("frame_hop_seconds", -1.0)), FRAME_HOP_SECONDS)
            or not math.isclose(float(grid.get("fixed_audio_seconds", -1.0)), FIXED_AUDIO_SECONDS)
        ):
            raise ValueError(f"dense index is not on the fixed 250x40ms grid: {index_path}")
        labels = [str(value) for value in payload.get("labels") or []]
        feature_dim = int(payload.get("feature_dim", -1))
        if not labels or feature_dim <= 0 or int(payload.get("num_labels", -1)) != len(labels):
            raise ValueError(f"invalid labels/feature_dim in {index_path}")
        if self.labels is None:
            self.labels = labels
            self.feature_dim = feature_dim
        elif self.labels != labels or self.feature_dim != feature_dim:
            raise ValueError("all dense indexes must have identical label order and feature width")
        shards = payload.get("shards")
        if not isinstance(shards, list) or len(shards) != int(payload.get("shard_count", -1)):
            raise ValueError(f"invalid shard table in {index_path}")
        seen_count = 0
        for expected_index, descriptor in enumerate(shards):
            if int(descriptor.get("index", -1)) != expected_index:
                raise ValueError(f"non-contiguous shard index in {index_path}")
            shard_path = _safe_shard_path(index_path, str(descriptor.get("path") or ""))
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            expected_sha = str(descriptor.get("sha256") or "")
            actual_sha = _sha256_file(shard_path)
            if len(expected_sha) != 64 or actual_sha != expected_sha:
                raise ValueError(f"shard sha256 mismatch: {shard_path}")
            shard = self._load_shard(shard_path)
            if shard.get("format") != DENSE_SHARD_FORMAT:
                raise ValueError(f"bad dense shard format: {shard_path}")
            scene_ids = shard.get("scene_ids")
            scenes = shard.get("scenes")
            events = shard.get("gold_events")
            features = shard.get("features")
            logits = shard.get("logits")
            valid_frames = shard.get("valid_frames")
            if not isinstance(scene_ids, list) or not scene_ids:
                raise ValueError(f"missing scene ids: {shard_path}")
            count = len(scene_ids)
            if int(descriptor.get("scene_count", -1)) != count:
                raise ValueError(f"shard scene count mismatch: {shard_path}")
            if not isinstance(scenes, list) or len(scenes) != count:
                raise ValueError(f"scene metadata mismatch: {shard_path}")
            if not isinstance(events, list) or len(events) != count:
                raise ValueError(f"event metadata mismatch: {shard_path}")
            if not isinstance(features, torch.Tensor) or tuple(features.shape) != (count, NUM_FRAMES, feature_dim):
                raise ValueError(f"feature tensor shape mismatch: {shard_path}")
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (count, NUM_FRAMES, len(labels)):
                raise ValueError(f"logit tensor shape mismatch: {shard_path}")
            if features.dtype != torch.float16 or logits.dtype != torch.float16:
                raise ValueError(f"dense feature/logit tensors must be float16: {shard_path}")
            if not isinstance(valid_frames, torch.Tensor) or tuple(valid_frames.shape) != (count,):
                raise ValueError(f"valid_frames shape mismatch: {shard_path}")
            if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
                raise ValueError(f"non-finite dense tensors: {shard_path}")
            for offset, raw_scene_id in enumerate(scene_ids):
                scene_id = str(raw_scene_id)
                if scene_id in self._locations:
                    raise ValueError(f"duplicate scene across dense indexes: {scene_id}")
                valid = int(valid_frames[offset])
                if valid < 1 or valid > NUM_FRAMES:
                    raise ValueError(f"invalid valid_frames for {scene_id}: {valid}")
                metadata = dict(scenes[offset])
                if str(metadata.get("scene_id") or scene_id) != scene_id:
                    raise ValueError(f"scene metadata id mismatch for {scene_id}")
                duration = min(max(float(metadata.get("duration_seconds") or 0.0), 0.0), FIXED_AUDIO_SECONDS)
                expected_valid = min(NUM_FRAMES, max(0, int(math.ceil(duration / FRAME_HOP_SECONDS - 1e-5))))
                if expected_valid != valid:
                    raise ValueError(
                        f"valid_frames does not match the fixed 40 ms grid for {scene_id}: "
                        f"stored={valid} expected={expected_valid}"
                    )
                metadata.update(
                    {
                        "scene_id": scene_id,
                        "events": list(events[offset]),
                        "gold_events": list(events[offset]),
                        "valid_frames": valid,
                    }
                )
                self._locations[scene_id] = (shard_path, offset)
                self._metadata[scene_id] = metadata
            seen_count += count
        if seen_count != int(payload.get("scene_count", -1)):
            raise ValueError(f"index scene count mismatch: {index_path}")
        self.index_sha256[str(index_path)] = _sha256_file(index_path)

    @property
    def scene_ids(self) -> set[str]:
        return set(self._locations)

    def metadata(self, scene_id: str) -> dict[str, Any]:
        try:
            return dict(self._metadata[scene_id])
        except KeyError as error:
            raise KeyError(f"scene not present in dense indexes: {scene_id}") from error

    def get(self, scene_id: str) -> dict[str, Any]:
        try:
            shard_path, offset = self._locations[scene_id]
        except KeyError as error:
            raise KeyError(f"scene not present in dense indexes: {scene_id}") from error
        shard = self._load_shard(shard_path)
        return {
            "features": shard["features"][offset],
            "logits": shard["logits"][offset],
            "valid_frames": int(shard["valid_frames"][offset]),
        }

    def preload(self, scene_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        return {scene_id: self.get(scene_id) for scene_id in scene_ids}


def _qa_interval(value: Any, context: str, *, allow_none: bool) -> tuple[float, float] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context} must be a two-value interval")
    onset, offset = float(value[0]), float(value[1])
    if (
        not math.isfinite(onset)
        or not math.isfinite(offset)
        or onset < 0.0
        or offset <= onset
        or offset > FIXED_AUDIO_SECONDS + 1e-8
    ):
        raise ValueError(f"{context} is outside the fixed 10 s task")
    return onset, offset


def load_explicit_qa_manifest(
    path: Path,
    *,
    allowed_scene_ids: Sequence[str],
) -> list[BalancedQAItem]:
    """Load the final builder's explicit headline QA subset.

    The explicit manifest is needed because selecting every adjacency would
    reintroduce a text-only answerability shortcut.  This loader independently
    checks the anchor+answer / anchor+verification evidence contract before a
    training sample can be created.
    """

    allowed = set(allowed_scene_ids)
    items: list[BalancedQAItem] = []
    seen_ids: set[str] = set()
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"non-object QA row at {path}:{line_number}")
            scene_id = str(row.get("scene_id") or "")
            if scene_id not in allowed:
                # A debug scene cap is allowed to select a strict subset of a
                # full explicit manifest.  Unknown ids are ignored here and
                # every retained id is still checked against the dense store.
                continue
            item_id = str(row.get("item_id") or "")
            if not item_id or item_id in seen_ids:
                raise ValueError(f"missing/duplicate item_id at {path}:{line_number}")
            seen_ids.add(item_id)
            relation = str(row.get("relation") or "")
            if relation not in ("before", "after"):
                raise ValueError(f"unsupported relation for {item_id}: {relation!r}")
            anchor_label = str(row.get("anchor_label") or "")
            ordinal = int(row.get("anchor_ordinal", 0))
            if not anchor_label or ordinal < 1 or ordinal > 10:
                raise ValueError(f"invalid anchor metadata for {item_id}")
            anchor = _qa_interval(
                row.get("gold_anchor_interval"), f"{item_id}.anchor", allow_none=False
            )
            assert anchor is not None
            answer = _qa_interval(
                row.get("gold_answer_interval"), f"{item_id}.answer", allow_none=True
            )
            verification = _qa_interval(
                row.get("gold_verification_interval"),
                f"{item_id}.verification",
                allow_none=True,
            )
            raw_evidence = row.get("gold_evidence_intervals")
            if not isinstance(raw_evidence, list) or not raw_evidence:
                raise ValueError(f"{item_id} has no evidence intervals")
            evidence = tuple(
                _qa_interval(value, f"{item_id}.evidence", allow_none=False)
                for value in raw_evidence
            )
            assert all(value is not None for value in evidence)
            evidence_intervals = tuple(value for value in evidence if value is not None)
            no_evidence = bool(row.get("no_evidence", False))
            answer_label_raw = row.get("answer_label")
            answer_label = None if answer_label_raw is None else str(answer_label_raw)
            if no_evidence:
                if answer is not None or answer_label is not None or verification is None:
                    raise ValueError(
                        f"{item_id} violates NONE=anchor+verification evidence policy"
                    )
                if len(evidence_intervals) != 1:
                    raise ValueError(f"{item_id} NONE evidence must be one verification window")
                expected_none_evidence = (
                    (verification[0], anchor[1])
                    if relation == "before"
                    else (anchor[0], verification[1])
                )
                if evidence_intervals != (expected_none_evidence,):
                    raise ValueError(
                        f"{item_id} NONE evidence is not anchor+complete verification region"
                    )
            else:
                if answer is None or not answer_label or verification is not None:
                    raise ValueError(
                        f"{item_id} violates positive=anchor+answer evidence policy"
                    )
                if evidence_intervals != (anchor, answer):
                    raise ValueError(f"{item_id} positive evidence is not anchor+answer")
            items.append(
                BalancedQAItem(
                    item_id=item_id,
                    scene_id=scene_id,
                    relation=relation,
                    question=str(row.get("question") or ""),
                    answer=str(row.get("answer") or "no_evidence"),
                    no_evidence=no_evidence,
                    no_evidence_reason=(
                        None
                        if row.get("no_evidence_reason") is None
                        else str(row.get("no_evidence_reason"))
                    ),
                    anchor_label=anchor_label,
                    anchor_ordinal=ordinal,
                    answer_label=answer_label,
                    gold_evidence_intervals=evidence_intervals,
                    gold_anchor_interval=anchor,
                    gold_answer_interval=answer,
                    gold_verification_interval=verification,
                    source_route=str(row.get("source_route") or "explicit_clean_scene_qa"),
                )
            )
    if not items:
        raise ValueError(f"explicit QA manifest has no rows for selected scenes: {path}")
    # The final builder pairs positive/NONE examples in every exact text-side
    # key.  Recheck the property here so a biased or hand-edited manifest
    # cannot silently become a paper training split.
    target_counts: dict[tuple[str, str, int], Counter[bool]] = defaultdict(Counter)
    for item in items:
        target_counts[(item.relation, item.anchor_label, item.anchor_ordinal)][
            item.no_evidence
        ] += 1
    imbalanced = [
        (*key, counts[False], counts[True])
        for key, counts in sorted(target_counts.items())
        if counts[False] != counts[True] or counts[False] == 0
    ]
    if imbalanced:
        raise ValueError(
            "explicit QA manifest is not positive/NONE-balanced by "
            f"(relation, anchor_label, ordinal): {imbalanced[:5]}"
        )
    return items


class DenseQADataset(Dataset[dict[str, Any]]):
    """Balanced temporal QA examples backed by verified dense features."""

    def __init__(
        self,
        store: DenseFeatureStore,
        scene_ids: Sequence[str],
        *,
        max_answerable_per_scene: int,
        max_no_evidence_per_scene: int,
        seed: int,
        preload: bool,
        explicit_qa_manifest: Path | None = None,
    ) -> None:
        missing = [scene_id for scene_id in scene_ids if scene_id not in store.scene_ids]
        if missing:
            raise ValueError(f"scene list contains ids absent from dense indexes: {missing[:5]}")
        if explicit_qa_manifest is None:
            rows = [store.metadata(scene_id) for scene_id in scene_ids]
            self.items = build_balanced_qa_items(
                rows,
                max_answerable_per_scene=max_answerable_per_scene,
                max_no_evidence_per_scene=max_no_evidence_per_scene,
                seed=seed,
            )
        else:
            self.items = load_explicit_qa_manifest(
                explicit_qa_manifest,
                allowed_scene_ids=scene_ids,
            )
        if not self.items:
            raise ValueError("balanced QA builder emitted no rows")
        self.store = store
        self.label_to_id = {label: index for index, label in enumerate(store.labels or [])}
        # An answerable target outside the ontology is a benchmark error, not
        # a no-evidence target.  Fail before the first optimizer step.
        unavailable = sorted(
            {
                str(item.answer_label)
                for item in self.items
                if not item.no_evidence and str(item.answer_label) not in self.label_to_id
            }
        )
        if unavailable:
            raise ValueError(f"answer labels unavailable from ontology: {unavailable[:10]}")
        bad_anchors = sorted({item.anchor_label for item in self.items if item.anchor_label not in self.label_to_id})
        if bad_anchors:
            raise ValueError(f"anchor labels unavailable from ontology: {bad_anchors[:10]}")
        used_scene_ids = list(dict.fromkeys(item.scene_id for item in self.items))
        self.preloaded = store.preload(used_scene_ids) if preload else None

    def __len__(self) -> int:
        return len(self.items)

    def _dense(self, scene_id: str) -> dict[str, Any]:
        return self.preloaded[scene_id] if self.preloaded is not None else self.store.get(scene_id)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        dense = self._dense(item.scene_id)
        valid_frames = int(dense["valid_frames"])
        valid_mask = torch.arange(NUM_FRAMES) < valid_frames
        anchor_mask = intervals_to_40ms_mask([item.gold_anchor_interval], NUM_FRAMES).bool() & valid_mask
        answer_intervals: list[tuple[float, float]] = []
        if item.gold_answer_interval is not None:
            answer_intervals.append(item.gold_answer_interval)
        answer_mask = intervals_to_40ms_mask(answer_intervals, NUM_FRAMES).bool() & valid_mask
        evidence_mask = intervals_to_40ms_mask(item.gold_evidence_intervals, NUM_FRAMES).bool() & valid_mask
        answer_label = -1 if item.no_evidence else self.label_to_id[str(item.answer_label)]
        relation_id = RELATION_BEFORE if item.relation == "before" else RELATION_AFTER
        return {
            "features": dense["features"].to(torch.float32),
            "detector_logits": dense["logits"].to(torch.float32),
            "valid_mask": valid_mask,
            "anchor_label": self.label_to_id[item.anchor_label],
            "relation_id": relation_id,
            "ordinal": min(int(item.anchor_ordinal), 10),
            "gold_answer_label": answer_label,
            "gold_anchor_mask": anchor_mask.to(torch.float32),
            "gold_answer_mask": answer_mask.to(torch.float32),
            "gold_evidence_mask": evidence_mask.to(torch.float32),
            "item_id": item.item_id,
            "scene_id": item.scene_id,
        }


def collate_dense_qa(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "features",
        "detector_logits",
        "valid_mask",
        "anchor_label",
        "relation_id",
        "ordinal",
        "gold_answer_label",
        "gold_anchor_mask",
        "gold_answer_mask",
        "gold_evidence_mask",
    )
    result: dict[str, Any] = {}
    for key in tensor_keys:
        values = [row[key] for row in rows]
        result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
    result["item_id"] = [str(row["item_id"]) for row in rows]
    result["scene_id"] = [str(row["scene_id"]) for row in rows]
    return result


def class_statistics(dataset: DenseQADataset, num_classes: int, *, beta: float, smoothing: float) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not 0.0 <= beta < 1.0:
        raise ValueError("class-balance beta must be in [0, 1)")
    if smoothing <= 0:
        raise ValueError("prior smoothing must be positive")
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for item in dataset.items:
        if not item.no_evidence:
            label = dataset.label_to_id[str(item.answer_label)]
            counts[label] += 1
    if counts.sum() == 0:
        raise ValueError("training split has no answerable QA rows")
    priors = (counts + smoothing) / (counts.sum() + smoothing * num_classes)
    effective = 1.0 - torch.pow(torch.full_like(counts, beta), counts.clamp_min(1.0))
    weights = (1.0 - beta) / effective.clamp_min(1e-12) if beta > 0 else torch.ones_like(counts)
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    nonzero = weights > 0
    weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    return priors.to(torch.float32), weights.to(torch.float32), [int(value) for value in counts.tolist()]


def _masked_balanced_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    error = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    valid_float = valid.to(error.dtype)
    positive = target * valid_float
    negative = (1.0 - target) * valid_float
    pos_count = positive.sum(dim=1)
    neg_count = negative.sum(dim=1)
    pos_loss = (error * positive).sum(dim=1) / pos_count.clamp_min(1.0)
    neg_loss = (error * negative).sum(dim=1) / neg_count.clamp_min(1.0)
    combined = torch.where(
        (pos_count > 0) & (neg_count > 0),
        0.5 * (pos_loss + neg_loss),
        torch.where(pos_count > 0, pos_loss, neg_loss),
    )
    return combined.mean()


def _masked_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid() * valid.to(logits.dtype)
    target = target * valid.to(target.dtype)
    intersection = (probability * target).sum(dim=1)
    denominator = probability.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def _balanced_verifier_bce(logit: torch.Tensor, answerable: torch.Tensor) -> torch.Tensor:
    error = F.binary_cross_entropy_with_logits(logit, answerable.to(logit.dtype), reduction="none")
    positive = answerable
    negative = ~answerable
    if bool(positive.any()) and bool(negative.any()):
        return 0.5 * (error[positive].mean() + error[negative].mean())
    return error.mean()


def training_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    log_prior: torch.Tensor,
    class_weights: torch.Tensor,
    tau: float,
) -> dict[str, torch.Tensor]:
    gold = batch["gold_answer_label"].long()
    answerable = gold >= 0
    answer_logits = outputs["answer_logits"]
    if bool(answerable.any()):
        selected_gold = gold[answerable]
        per_row = F.cross_entropy(
            answer_logits[answerable] + tau * log_prior[None, :],
            selected_gold,
            reduction="none",
        )
        selected_weight = class_weights[selected_gold]
        label_loss = (per_row * selected_weight).sum() / selected_weight.sum().clamp_min(1e-6)
    else:
        label_loss = answer_logits.sum() * 0.0
    valid = batch["valid_mask"].bool()
    anchor_bce = _masked_balanced_bce(outputs["anchor_mask_logits"], batch["gold_anchor_mask"], valid)
    anchor_dice = _masked_dice(outputs["anchor_mask_logits"], batch["gold_anchor_mask"], valid)
    answer_bce = _masked_balanced_bce(outputs["answer_mask_logits"], batch["gold_answer_mask"], valid)
    answer_dice = _masked_dice(outputs["answer_mask_logits"], batch["gold_answer_mask"], valid)
    verifier = _balanced_verifier_bce(outputs["answerability_logit"], answerable)
    probability = outputs["answer_mask_logits"].sigmoid()
    outside = (1.0 - outputs["relation_mask"].clamp(0.0, 1.0)) * valid.to(probability.dtype)
    relation = (probability * outside).sum(dim=1).div((probability * valid).sum(dim=1).clamp_min(1e-6)).mean()
    total = label_loss + 0.75 * verifier + anchor_bce + anchor_dice + answer_bce + answer_dice + 0.2 * relation
    return {
        "loss": total,
        "label": label_loss,
        "answerability": verifier,
        "anchor_bce": anchor_bce,
        "anchor_dice": anchor_dice,
        "answer_bce": answer_bce,
        "answer_dice": answer_dice,
        "relation": relation,
    }


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _mask_padding(batch: dict[str, Any]) -> None:
    invalid = ~batch["valid_mask"].bool()
    batch["features"] = batch["features"].masked_fill(invalid[:, :, None], 0.0)
    batch["detector_logits"] = batch["detector_logits"].masked_fill(invalid[:, :, None], -100.0)


def _binary_iou(predicted: torch.Tensor, gold: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    predicted = predicted & valid
    gold = gold & valid
    intersection = (predicted & gold).sum(dim=1).to(torch.float32)
    union = (predicted | gold).sum(dim=1).to(torch.float32)
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


@torch.no_grad()
def evaluate(
    model: DenseTemporalReasoner,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    mask_threshold: float,
    answerability_threshold: float | None,
) -> dict[str, Any]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    label_predictions: list[torch.Tensor] = []
    gold_labels: list[torch.Tensor] = []
    anchor_ious: list[torch.Tensor] = []
    answer_ious: list[torch.Tensor] = []
    evidence_ious: list[torch.Tensor] = []
    topk_hits: dict[int, list[torch.Tensor]] = {1: [], 5: [], 10: []}
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        _mask_padding(batch)
        outputs = model(
            batch["features"],
            batch["detector_logits"],
            batch["anchor_label"].long(),
            batch["relation_id"].long(),
            batch["ordinal"].long(),
            teacher_forcing=0.0,
            valid_frame_mask=batch["valid_mask"],
        )
        valid = batch["valid_mask"].bool()
        anchor_pred = (outputs["anchor_mask_logits"].sigmoid() >= mask_threshold) & valid
        answer_pred = (outputs["answer_mask_logits"].sigmoid() >= mask_threshold) & valid
        anchor_iou = _binary_iou(anchor_pred, batch["gold_anchor_mask"].bool(), valid)
        gold = batch["gold_answer_label"].long()
        answerable = gold >= 0
        answer_iou = _binary_iou(answer_pred, batch["gold_answer_mask"].bool(), valid)
        relation_pred = (outputs["relation_mask"] >= mask_threshold) & valid
        # NONE evidence verifies absence: predicted anchor plus the relation
        # search region.  Positive evidence is anchor plus answer.
        # We retain both variants until the calibrated NONE decision is known.
        positive_evidence = anchor_pred | answer_pred
        negative_evidence = anchor_pred | relation_pred
        evidence_positive_iou = _binary_iou(positive_evidence, batch["gold_evidence_mask"].bool(), valid)
        evidence_negative_iou = _binary_iou(negative_evidence, batch["gold_evidence_mask"].bool(), valid)

        probabilities.append(outputs["answerability_logit"].sigmoid().cpu())
        label_predictions.append(outputs["answer_logits"].argmax(dim=1).cpu())
        gold_labels.append(gold.cpu())
        anchor_ious.append(anchor_iou.cpu())
        answer_ious.append(answer_iou.cpu())
        # Store two columns; thresholding chooses the applicable evidence row.
        evidence_ious.append(torch.stack((evidence_positive_iou, evidence_negative_iou), dim=1).cpu())
        if bool(answerable.any()):
            for k in topk_hits:
                actual_k = min(k, outputs["answer_logits"].shape[1])
                hit = (outputs["answer_logits"][answerable].topk(actual_k, dim=1).indices == gold[answerable, None]).any(dim=1)
                topk_hits[k].append(hit.cpu())

    probability = torch.cat(probabilities)
    label_prediction = torch.cat(label_predictions)
    gold = torch.cat(gold_labels)
    anchor_iou = torch.cat(anchor_ious)
    answer_iou = torch.cat(answer_ious)
    evidence_pair = torch.cat(evidence_ious)
    answerable = gold >= 0

    def metrics_at(threshold: float) -> dict[str, Any]:
        predicted_answerable = probability >= threshold
        answer_correct = predicted_answerable & answerable & (label_prediction == gold)
        answerable_accuracy = float(answer_correct[answerable].float().mean()) if bool(answerable.any()) else 0.0
        no_evidence = ~answerable
        no_evidence_accuracy = float((~predicted_answerable[no_evidence]).float().mean()) if bool(no_evidence.any()) else 0.0
        balanced = 0.5 * (answerable_accuracy + no_evidence_accuracy)
        selected_evidence_iou = torch.where(predicted_answerable, evidence_pair[:, 0], evidence_pair[:, 1])
        strict = answer_correct & (anchor_iou >= 0.30) & (answer_iou >= 0.30)
        return {
            "answerability_threshold": float(threshold),
            "direct_answerable_accuracy": answerable_accuracy,
            "no_evidence_accuracy": no_evidence_accuracy,
            "balanced_accuracy": balanced,
            "overall_answer_accuracy": float((answer_correct | ((~predicted_answerable) & no_evidence)).float().mean()),
            "anchor_iou": float(anchor_iou.mean()),
            "answer_iou": float(answer_iou[answerable].mean()) if bool(answerable.any()) else 0.0,
            "union_evidence_iou": float(selected_evidence_iou.mean()),
            "union_evidence_iou_answerable": float(selected_evidence_iou[answerable].mean()) if bool(answerable.any()) else 0.0,
            "strict_joint_answer_anchor_answer_iou_0.30": float(strict[answerable].float().mean()) if bool(answerable.any()) else 0.0,
            "answerable_count": int(answerable.sum()),
            "no_evidence_count": int((~answerable).sum()),
        }

    if answerability_threshold is None:
        candidates = [index / 100.0 for index in range(1, 100)]
        scored = [metrics_at(threshold) for threshold in candidates]
        metrics = max(scored, key=lambda row: (row["balanced_accuracy"], -abs(row["answerability_threshold"] - 0.5)))
        metrics["threshold_calibrated_on_this_split"] = True
    else:
        metrics = metrics_at(answerability_threshold)
        metrics["threshold_calibrated_on_this_split"] = False
    for k, values in topk_hits.items():
        metrics[f"answer_label_top{k}"] = float(torch.cat(values).float().mean()) if values else 0.0
    return metrics


def teacher_forcing_for_epoch(stage: str, epoch_index: int, stage_epochs: int, start: float, end: float) -> float:
    if stage == "A":
        return 1.0
    if stage_epochs <= 1:
        return float(end)
    progress = epoch_index / float(stage_epochs - 1)
    return float(start + progress * (end - start))


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise SystemExit("batch-size must be positive and num-workers non-negative")
    if args.epochs_stage_a < 0 or args.epochs_stage_b < 0 or args.epochs_stage_a + args.epochs_stage_b < 1:
        raise SystemExit("at least one stage epoch is required")
    for value, name in ((args.stage_b_teacher_start, "stage-b-teacher-start"), (args.stage_b_teacher_end, "stage-b-teacher-end")):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be in [0, 1]")
    if (args.train_qa_manifest is None) != (args.dev_qa_manifest is None):
        raise SystemExit(
            "--train-qa-manifest and --dev-qa-manifest must be supplied together"
        )
    output_dir = args.output_dir.resolve()
    receipt_path = output_dir / "receipt.json"
    checkpoint_path = output_dir / "qdor_best.pt"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_gate = load_oracle_gate(
        args.oracle_gate_receipt,
        minimum_top1=args.min_oracle_answer_top1,
        allow_failed=args.allow_failed_oracle_gate,
    )

    _seed_everything(args.seed)
    train_ids = load_scene_list(args.train_scene_list)
    dev_ids = load_scene_list(args.dev_scene_list)
    if args.max_train_scenes > 0:
        train_ids = train_ids[: args.max_train_scenes]
    if args.max_dev_scenes > 0:
        dev_ids = dev_ids[: args.max_dev_scenes]
    overlap = sorted(set(train_ids) & set(dev_ids))
    if overlap:
        raise ValueError(f"train/dev scene lists overlap: {overlap[:10]}")

    store = DenseFeatureStore(args.dense_index, cache_size=args.shard_cache_size)
    train_dataset = DenseQADataset(
        store,
        train_ids,
        max_answerable_per_scene=args.max_answerable_per_scene,
        max_no_evidence_per_scene=args.max_no_evidence_per_scene,
        seed=args.seed,
        preload=args.preload,
        explicit_qa_manifest=args.train_qa_manifest,
    )
    dev_dataset = DenseQADataset(
        store,
        dev_ids,
        max_answerable_per_scene=args.max_answerable_per_scene,
        max_no_evidence_per_scene=args.max_no_evidence_per_scene,
        seed=args.seed,
        preload=args.preload,
        explicit_qa_manifest=args.dev_qa_manifest,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_dense_qa,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_dense_qa,
    )
    labels = store.labels or []
    priors, class_weights, class_counts = class_statistics(
        train_dataset,
        len(labels),
        beta=args.class_balance_beta,
        smoothing=args.prior_smoothing,
    )
    config = DenseTemporalReasonerConfig(
        feature_dim=int(store.feature_dim or 0),
        num_classes=len(labels),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    device = _device(args.device)
    model = DenseTemporalReasoner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    log_prior = priors.log().to(device)
    class_weights_device = class_weights.to(device)

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_metrics: dict[str, Any] | None = None
    global_epoch = 0
    stages = (("A", args.epochs_stage_a), ("B", args.epochs_stage_b))
    for stage, stage_epochs in stages:
        for stage_epoch in range(stage_epochs):
            global_epoch += 1
            teacher_forcing = teacher_forcing_for_epoch(
                stage,
                stage_epoch,
                stage_epochs,
                args.stage_b_teacher_start,
                args.stage_b_teacher_end,
            )
            model.train()
            totals: Counter[str] = Counter()
            examples = 0
            for raw_batch in train_loader:
                batch = _to_device(raw_batch, device)
                _mask_padding(batch)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    batch["features"],
                    batch["detector_logits"],
                    batch["anchor_label"].long(),
                    batch["relation_id"].long(),
                    batch["ordinal"].long(),
                    gold_anchor_mask=batch["gold_anchor_mask"],
                    gold_answer_label=batch["gold_answer_label"].long(),
                    teacher_forcing=teacher_forcing,
                    valid_frame_mask=batch["valid_mask"],
                )
                losses = training_loss(
                    outputs,
                    batch,
                    log_prior=log_prior,
                    class_weights=class_weights_device,
                    tau=args.logit_adjustment_tau,
                )
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"non-finite loss at epoch {global_epoch}")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                count = int(batch["features"].shape[0])
                examples += count
                for name, value in losses.items():
                    totals[name] += float(value.detach()) * count
            train_metrics = {name: value / max(examples, 1) for name, value in totals.items()}
            dev_metrics = evaluate(
                model,
                dev_loader,
                device,
                mask_threshold=args.mask_threshold,
                answerability_threshold=None,
            )
            epoch_row = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch + 1,
                "teacher_forcing": teacher_forcing,
                "train": train_metrics,
                "dev": dev_metrics,
            }
            history.append(epoch_row)
            print(json.dumps(epoch_row, ensure_ascii=False, sort_keys=True), flush=True)
            score = float(dev_metrics["balanced_accuracy"])
            if score > best_score:
                best_score = score
                best_metrics = dict(dev_metrics)
                _atomic_torch(
                    {
                        "format": "qces_qdor_dense_checkpoint_v1",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "labels": labels,
                        "answerability_threshold": dev_metrics["answerability_threshold"],
                        "mask_threshold": args.mask_threshold,
                        "epoch": global_epoch,
                        "stage": stage,
                        "metrics": dev_metrics,
                        "class_priors": priors,
                        "class_weights": class_weights,
                    },
                    checkpoint_path,
                )
            partial = {
                "format": RECEIPT_FORMAT,
                "status": "running",
                "history": history,
                "best_dev": best_metrics,
            }
            _atomic_json(partial, output_dir / "training_history.json")

    assert best_metrics is not None
    receipt = {
        "format": RECEIPT_FORMAT,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Q-DOR dense onset reasoner",
        "data_protocol": {
            "fixed_frame_hop_seconds": FRAME_HOP_SECONDS,
            "num_frames": NUM_FRAMES,
            "train_scene_list": str(args.train_scene_list.resolve()),
            "dev_scene_list": str(args.dev_scene_list.resolve()),
            "train_scene_count_requested": len(train_ids),
            "dev_scene_count_requested": len(dev_ids),
            "scene_id_overlap": 0,
            "qa_source": (
                "explicit_conditionally_balanced_manifests"
                if args.train_qa_manifest is not None
                else "legacy_balanced_adjacency_builder"
            ),
            "train_qa_manifest": (
                {
                    "path": str(args.train_qa_manifest.resolve()),
                    "sha256": _sha256_file(args.train_qa_manifest.resolve()),
                }
                if args.train_qa_manifest is not None
                else None
            ),
            "dev_qa_manifest": (
                {
                    "path": str(args.dev_qa_manifest.resolve()),
                    "sha256": _sha256_file(args.dev_qa_manifest.resolve()),
                }
                if args.dev_qa_manifest is not None
                else None
            ),
            "balanced_builder": {
                "max_answerable_per_scene": args.max_answerable_per_scene,
                "max_no_evidence_per_scene": args.max_no_evidence_per_scene,
                "min_onset_gap_seconds": 0.08,
                "require_non_overlapping": True,
                "exclude_same_label_pairs": True,
            },
            "dense_indexes": store.index_sha256,
            "oracle_representation_gate": oracle_gate,
        },
        "dataset": {
            "train_qa_count": len(train_dataset),
            "dev_qa_count": len(dev_dataset),
            "train_answerable_count": sum(not item.no_evidence for item in train_dataset.items),
            "train_no_evidence_count": sum(item.no_evidence for item in train_dataset.items),
            "dev_answerable_count": sum(not item.no_evidence for item in dev_dataset.items),
            "dev_no_evidence_count": sum(item.no_evidence for item in dev_dataset.items),
            "num_classes": len(labels),
            "answer_class_counts": class_counts,
        },
        "model_config": asdict(config),
        "optimization": {
            "epochs_stage_a": args.epochs_stage_a,
            "epochs_stage_b": args.epochs_stage_b,
            "stage_b_teacher_start": args.stage_b_teacher_start,
            "stage_b_teacher_end": args.stage_b_teacher_end,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "logit_adjustment_tau": args.logit_adjustment_tau,
            "class_balance_beta": args.class_balance_beta,
            "prior_smoothing": args.prior_smoothing,
            "none_loss": "separate batch-balanced binary verifier",
            "answer_loss": "answerable-only class-balanced logit-adjusted cross entropy",
        },
        "selection_metric": "dev balanced_accuracy (threshold calibrated on dev)",
        "best_dev": best_metrics,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    _atomic_json(receipt, receipt_path)
    return receipt


if __name__ == "__main__":
    main()
