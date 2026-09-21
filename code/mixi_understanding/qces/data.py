"""Dataset bridge across QA-removal v2 and QCES v3/v4/v5 manifests."""

from __future__ import annotations

import json
import hashlib
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from mixi_understanding.data.qa_schema import Interval, QARecord, parse_record
from mixi_understanding.data.qces_schema import (
    QCESRecord,
    SCHEMA_VERSION as QCES_V3_SCHEMA_VERSION,
    parse_qces_record,
)
from mixi_understanding.data.qces_v4_schema import (
    QCESV4Record,
    SCHEMA_VERSION as QCES_V4_SCHEMA_VERSION,
    parse_qces_v4_record,
)
from mixi_understanding.data.qces_v5_schema import (
    DERIVED_SCHEMA_VERSION as QCES_V5_DERIVED_SCHEMA_VERSION,
    DERIVED_STORAGE_MODE,
    QCESV5Record,
    SCHEMA_VERSION as QCES_V5_SCHEMA_VERSION,
    parse_qces_v5_record,
)
from mixi_understanding.qces.tokenization import StableHashTokenizer

if TYPE_CHECKING:
    from mixi_understanding.qces.counterfactual import CounterfactualGroupPlan


ManifestRecord = Union[QARecord, QCESRecord, QCESV4Record, QCESV5Record]
ExplicitStemRecord = Union[QCESRecord, QCESV4Record, QCESV5Record]


@dataclass
class QCESExample:
    sample_id: str
    question: str
    answer: str
    mixture: torch.Tensor
    evidence: torch.Tensor
    residual: torch.Tensor
    anchor_stem: torch.Tensor
    answer_stem: torch.Tensor
    anchor_mask: torch.Tensor
    answer_mask: torch.Tensor
    no_evidence: torch.Tensor


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise TypeError(f"expected an object at {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _read_mono(path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        raise ValueError(
            f"expected mono audio at {path}, got {waveform.shape[1]} channels"
        )
    waveform = np.ascontiguousarray(waveform[:, 0])
    if not np.isfinite(waveform).all():
        raise ValueError(f"audio contains NaN/Inf: {path}")
    return torch.from_numpy(waveform), int(sample_rate)


def _paint_intervals(
    intervals: Iterable[Interval], sample_rate: int, num_samples: int
) -> torch.Tensor:
    mask = torch.zeros(num_samples, dtype=torch.float32)
    for onset, offset in intervals:
        start = max(0, min(int(round(onset * sample_rate)), num_samples))
        end = max(start, min(int(round(offset * sample_rate)), num_samples))
        mask[start:end] = 1.0
    return mask


class QCESManifestDataset(Dataset[QCESExample]):
    """Load a validated manifest and optionally draw evidence-focused crops.

    QA-removal v2 and QCES v3/v4 retain their historical eager behavior.
    Scene/event-derived QCES v5 is decoded lazily with a bounded scene cache.
    """

    def __init__(
        self,
        manifest_path: Path,
        crop_samples: Optional[int] = None,
        random_crop: bool = True,
        seed: int = 2026,
        derived_scene_cache_size: int = 1,
        align_v5_family_crops: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.dataset_root = self.manifest_path.parent
        self.crop_samples = crop_samples
        self.random_crop = random_crop
        self.seed = seed
        self.random = random.Random(seed)
        self.align_v5_family_crops = align_v5_family_crops
        if derived_scene_cache_size < 0:
            raise ValueError("derived_scene_cache_size must be non-negative")
        self.derived_scene_cache_size = derived_scene_cache_size
        rows = read_jsonl(self.manifest_path)
        schema_versions = {row.get("schema_version") for row in rows}
        if schema_versions == {QCES_V3_SCHEMA_VERSION}:
            self.records: List[ManifestRecord] = [
                parse_qces_record(row) for row in rows
            ]
        elif schema_versions == {QCES_V4_SCHEMA_VERSION}:
            self.records = [parse_qces_v4_record(row) for row in rows]
        elif schema_versions in (
            {QCES_V5_SCHEMA_VERSION},
            {QCES_V5_DERIVED_SCHEMA_VERSION},
        ):
            self.records = [parse_qces_v5_record(row) for row in rows]
        else:
            self.records = [parse_record(row) for row in rows]
        self._lazy_derived_v5 = bool(self.records) and all(
            isinstance(record, QCESV5Record)
            and record.storage_mode == DERIVED_STORAGE_MODE
            for record in self.records
        )
        # Legacy manifests retain eager behavior.  Paper-scale derived v5 is
        # loaded lazily and keeps only a bounded number of complete scenes per
        # DataLoader worker.
        self._audio_cache: Dict[str, tuple[torch.Tensor, int]] = {}
        self._derived_scene_cache: OrderedDict[
            str, Dict[str, tuple[torch.Tensor, int]]
        ] = OrderedDict()
        self.examples: Optional[List[QCESExample]] = (
            None
            if self._lazy_derived_v5
            else [self._load(record) for record in self.records]
        )
        sample_rates = {record.sample_rate for record in self.records}
        if len(sample_rates) != 1:
            raise ValueError(
                f"all examples must share one sample rate, got {sample_rates}"
            )
        self.sample_rate = next(iter(sample_rates))
        if crop_samples is not None:
            if crop_samples <= 0:
                raise ValueError("crop_samples must be positive")
            shortest = min(record.num_samples for record in self.records)
            if crop_samples > shortest:
                raise ValueError(
                    f"crop_samples={crop_samples} exceeds shortest clip={shortest}"
                )
            for index, record in enumerate(self.records):
                example = None if self.examples is None else self.examples[index]
                span = self._required_crop_span(example, record)
                if span is None:
                    continue
                evidence_span = span[1] - span[0]
                if evidence_span > crop_samples:
                    raise ValueError(
                        f"crop_samples={crop_samples} cannot preserve both roles for "
                        f"{record.sample_id}; evidence span={evidence_span}. Increase "
                        "--crop-seconds so relational supervision remains valid."
                    )
        self._aligned_crop_starts: Dict[str, int] = {}
        if align_v5_family_crops:
            if not all(isinstance(record, QCESV5Record) for record in self.records):
                raise ValueError(
                    "family-aligned crops require a pure QCES-v5 manifest"
                )
            if crop_samples is not None:
                self._aligned_crop_starts = self._build_aligned_v5_crop_starts()

    def _resolve_audio(self, relative_path: str) -> Path:
        path = (self.dataset_root / relative_path).resolve()
        if self.dataset_root not in path.parents:
            raise ValueError(f"audio path escapes dataset root: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _read_audio(self, relative_path: str) -> tuple[torch.Tensor, int]:
        if relative_path not in self._audio_cache:
            self._audio_cache[relative_path] = _read_mono(
                self._resolve_audio(relative_path)
            )
        return self._audio_cache[relative_path]

    def _read_derived_scene(
        self, record: QCESV5Record
    ) -> Dict[str, tuple[torch.Tensor, int]]:
        cached = self._derived_scene_cache.get(record.scene_id)
        if cached is not None:
            self._derived_scene_cache.move_to_end(record.scene_id)
            return cached
        paths = [record.mixture_path, *(event.stem_path for event in record.events)]
        loaded = {
            relative: _read_mono(self._resolve_audio(relative))
            for relative in paths
        }
        if self.derived_scene_cache_size > 0:
            self._derived_scene_cache[record.scene_id] = loaded
            while len(self._derived_scene_cache) > self.derived_scene_cache_size:
                self._derived_scene_cache.popitem(last=False)
        return loaded

    def _load(self, record: ManifestRecord) -> QCESExample:
        if isinstance(record, QCESV5Record) and record.storage_mode == DERIVED_STORAGE_MODE:
            return self._load_v5_derived(record)
        if isinstance(record, (QCESRecord, QCESV4Record, QCESV5Record)):
            return self._load_explicit_stems(record)
        return self._load_v2(record)

    def _validate_loaded(
        self,
        record: ManifestRecord,
        loaded: Sequence[tuple[torch.Tensor, int]],
    ) -> tuple[torch.Tensor, ...]:
        if not loaded:
            raise ValueError("at least one waveform is required")
        rates = {sample_rate for _, sample_rate in loaded}
        shapes = {waveform.numel() for waveform, _ in loaded}
        if rates != {record.sample_rate}:
            raise ValueError(f"sample-rate mismatch for {record.sample_id}: {rates}")
        if shapes != {record.num_samples}:
            raise ValueError(
                f"waveform length mismatch for {record.sample_id}: {shapes}"
            )
        return tuple(waveform for waveform, _ in loaded)

    def _load_v2(self, record: QARecord) -> QCESExample:
        paths = (
            record.mixture_path,
            record.clean_path,
            record.interference_stem_path,
        )
        loaded = [self._read_audio(path) for path in paths]
        mixture, evidence, residual = self._validate_loaded(record, loaded)
        anchor_mask = _paint_intervals(
            record.anchor_intervals, record.sample_rate, record.num_samples
        )
        answer_mask = _paint_intervals(
            record.answer_intervals, record.sample_rate, record.num_samples
        )
        return QCESExample(
            sample_id=record.sample_id,
            question=record.question,
            answer=record.answer,
            mixture=mixture,
            evidence=evidence,
            residual=residual,
            anchor_stem=evidence * anchor_mask,
            answer_stem=evidence * answer_mask,
            anchor_mask=anchor_mask,
            answer_mask=answer_mask,
            no_evidence=torch.tensor(0.0),
        )

    def _load_explicit_stems(self, record: ExplicitStemRecord) -> QCESExample:
        role_paths = (
            record.evidence_stem_path,
            record.residual_stem_path,
            record.anchor_stem_path,
            record.answer_stem_path,
        )
        if any(path is None for path in role_paths):
            raise ValueError("materialized QCES record lacks a role-stem path")
        paths = (record.mixture_path, *(str(path) for path in role_paths))
        mixture, evidence, residual, anchor_stem, answer_stem = self._validate_loaded(
            record, [self._read_audio(path) for path in paths]
        )
        return QCESExample(
            sample_id=record.sample_id,
            question=record.question,
            answer=record.answer,
            mixture=mixture,
            evidence=evidence,
            residual=residual,
            anchor_stem=anchor_stem,
            answer_stem=answer_stem,
            anchor_mask=_paint_intervals(
                record.anchor_intervals, record.sample_rate, record.num_samples
            ),
            answer_mask=_paint_intervals(
                record.answer_intervals, record.sample_rate, record.num_samples
            ),
            no_evidence=torch.tensor(float(record.no_evidence)),
        )

    def _load_v5_derived(self, record: QCESV5Record) -> QCESExample:
        scene_audio = self._read_derived_scene(record)
        mixture_loaded = scene_audio[record.mixture_path]
        event_loaded = {
            event.event_id: scene_audio[event.stem_path]
            for event in record.events
        }
        validated = self._validate_loaded(
            record, [mixture_loaded, *event_loaded.values()]
        )
        mixture = validated[0]
        event_audio = {
            event_id: validated[index + 1]
            for index, event_id in enumerate(event_loaded)
        }

        def role_sum(event_ids: Sequence[str]) -> torch.Tensor:
            result = torch.zeros_like(mixture)
            for event_id in event_ids:
                result = result + event_audio[event_id]
            return result

        evidence = role_sum(record.evidence_event_ids)
        return QCESExample(
            sample_id=record.sample_id,
            question=record.question,
            answer=record.answer,
            mixture=mixture,
            evidence=evidence,
            residual=mixture - evidence,
            anchor_stem=role_sum(record.anchor_event_ids),
            answer_stem=role_sum(record.answer_event_ids),
            anchor_mask=_paint_intervals(
                record.anchor_intervals, record.sample_rate, record.num_samples
            ),
            answer_mask=_paint_intervals(
                record.answer_intervals, record.sample_rate, record.num_samples
            ),
            no_evidence=torch.tensor(float(record.no_evidence)),
        )

    def __len__(self) -> int:
        # Derived v5 intentionally keeps ``examples=None`` and renders role
        # stems lazily.  Dataset cardinality always comes from parsed records.
        return len(self.records)

    def _crop_bounds(
        self, example: Optional[QCESExample], record: ManifestRecord
    ) -> tuple[int, int]:
        if self.crop_samples is None:
            raise RuntimeError("crop bounds require crop_samples")
        last = record.num_samples - self.crop_samples
        if last <= 0:
            return 0, 0
        span = self._required_crop_span(example, record)
        if span is None:
            return 0, last
        first, final = span
        low = max(0, final - self.crop_samples)
        high = min(first, last)
        if low > high:
            raise RuntimeError("validated evidence span no longer fits in the crop")
        return low, high

    def _build_aligned_v5_crop_starts(self) -> Dict[str, int]:
        """Choose one absolute crop start for every record in a v5 family.

        Surface controls, intervention variants, and same-scene question pairs
        must use identical sample coordinates.  Otherwise an apparent paired
        effect can be caused by the random crop rather than the declared
        intervention.
        """

        by_family: Dict[str, List[tuple[int, QCESV5Record]]] = {}
        for index, raw_record in enumerate(self.records):
            if not isinstance(raw_record, QCESV5Record):
                raise RuntimeError("aligned v5 crop received a non-v5 record")
            by_family.setdefault(raw_record.scene_family_id, []).append(
                (index, raw_record)
            )
        starts: Dict[str, int] = {}
        for family_id, indexed_records in sorted(by_family.items()):
            bounds = [
                self._crop_bounds(
                    None if self.examples is None else self.examples[index],
                    record,
                )
                for index, record in indexed_records
            ]
            low = max(bound[0] for bound in bounds)
            high = min(bound[1] for bound in bounds)
            if low > high:
                raise ValueError(
                    "counterfactual family has no common crop preserving its "
                    f"declared evidence roles: {family_id}"
                )
            if self.random_crop:
                width = high - low + 1
                digest = hashlib.sha256(
                    f"qces-v5-family-crop\0{self.seed}\0{family_id}".encode(
                        "utf-8"
                    )
                ).digest()
                start = low + int.from_bytes(digest[:8], "big") % width
            else:
                start = (low + high) // 2
            for _, record in indexed_records:
                starts[record.sample_id] = start
        return starts

    @staticmethod
    def _required_crop_span(
        example: Optional[QCESExample], record: ManifestRecord
    ) -> Optional[tuple[int, int]]:
        if isinstance(record, (QCESRecord, QCESV4Record, QCESV5Record)):
            semantic = [
                event for event in record.events if event.event_kind == "semantic"
            ]
            return (
                int(
                    round(
                        min(event.onset_seconds for event in semantic)
                        * record.sample_rate
                    )
                ),
                int(
                    round(
                        max(event.offset_seconds for event in semantic)
                        * record.sample_rate
                    )
                ),
            )
        if example is None:
            raise RuntimeError("legacy crop validation requires a loaded example")
        evidence_indices = torch.nonzero(
            (example.anchor_mask + example.answer_mask) > 0, as_tuple=False
        ).flatten()
        if evidence_indices.numel() == 0:
            return None
        return int(evidence_indices[0]), int(evidence_indices[-1]) + 1

    def _crop_start(
        self,
        example: QCESExample,
        record: ManifestRecord,
    ) -> int:
        assert self.crop_samples is not None
        if record.sample_id in self._aligned_crop_starts:
            return self._aligned_crop_starts[record.sample_id]
        low, high = self._crop_bounds(example, record)
        if low == high:
            return low
        if self.random_crop and isinstance(record, QCESV5Record):
            width = high - low + 1
            digest = hashlib.sha256(
                f"qces-v5-crop\0{self.seed}\0{record.sample_id}".encode("utf-8")
            ).digest()
            return low + int.from_bytes(digest[:8], "big") % width
        if self.random_crop and not isinstance(record, (QCESRecord, QCESV4Record)):
            return self.random.randint(low, high)
        return (low + high) // 2

    def __getitem__(self, index: int) -> QCESExample:
        example = (
            self._load(self.records[index])
            if self.examples is None
            else self.examples[index]
        )
        if self.crop_samples is None:
            return example
        start = self._crop_start(example, self.records[index])
        stop = start + self.crop_samples
        return QCESExample(
            sample_id=example.sample_id,
            question=example.question,
            answer=example.answer,
            mixture=example.mixture[start:stop],
            evidence=example.evidence[start:stop],
            residual=example.residual[start:stop],
            anchor_stem=example.anchor_stem[start:stop],
            answer_stem=example.answer_stem[start:stop],
            anchor_mask=example.anchor_mask[start:stop],
            answer_mask=example.answer_mask[start:stop],
            no_evidence=example.no_evidence,
        )


def collate_qces(
    examples: Sequence[QCESExample],
    tokenizer: StableHashTokenizer,
    counterfactual_plan: Optional["CounterfactualGroupPlan"] = None,
) -> Dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    lengths = {example.mixture.numel() for example in examples}
    if len(lengths) != 1:
        raise ValueError(
            "batch waveforms have different lengths; configure crop_samples"
        )
    questions = [example.question for example in examples]
    tokens = tokenizer.batch_encode(questions)
    batch = {
        "sample_ids": [example.sample_id for example in examples],
        "questions": questions,
        "answers": [example.answer for example in examples],
        "mixture": torch.stack([example.mixture for example in examples]),
        "evidence": torch.stack([example.evidence for example in examples]),
        "residual": torch.stack([example.residual for example in examples]),
        "anchor_stem": torch.stack([example.anchor_stem for example in examples]),
        "answer_stem": torch.stack([example.answer_stem for example in examples]),
        "anchor_mask": torch.stack([example.anchor_mask for example in examples]),
        "answer_mask": torch.stack([example.answer_mask for example in examples]),
        "no_evidence": torch.stack([example.no_evidence for example in examples]),
        "question_ids": tokens.input_ids,
        "question_mask": tokens.attention_mask,
    }
    if counterfactual_plan is not None:
        batch["counterfactual_groups"] = counterfactual_plan.batch_groups(
            batch["sample_ids"]
        )
    return batch
