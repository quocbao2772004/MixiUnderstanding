#!/usr/bin/env python3
"""Cache oracle AudioSep CLAP conditions used only as training supervision."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.data.qces_schema import QCESRecord
from mixi_understanding.data.qces_v4_schema import QCESV4Record
from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.config import (
    DUAL_ROLE_SEMANTIC_MODE,
    SEMANTIC_SEPARATION_MODES,
    UNION_SINGLE_SEMANTIC_MODE,
)
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    describe,
    oracle_prompt,
)
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    configure_determinism,
    load_frozen_encoder,
    source_tree_identity,
)


SUPPORTED_RECORD_TYPES = (QCESRecord, QCESV4Record, QCESV5Record)
ROLE_SEMANTIC_CACHE_FORMAT = "qces_audiosep_role_semantic_targets_v1"
ROLE_SEMANTIC_TARGET_SCOPE = "training_supervision_only"
ROLE_SEMANTIC_PROMPT_SOURCE = "anchor_and_answer_role_event_labels"
V5_UNION_SEMANTIC_CACHE_FORMAT = "qces_audiosep_union_semantic_targets_v2"
V5_UNION_SEMANTIC_TARGET_SCOPE = "training_supervision_only"
V5_UNION_SEMANTIC_PROMPT_SOURCE = (
    "unique_anchor_answer_role_labels_in_timeline_order"
)


def encode_prompts_batched(
    repository_root: Path,
    checkpoint_path: Path,
    prompts: Iterable[str],
    *,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Encode unique supervision prompts with bounded memory and exact state."""

    if batch_size <= 0:
        raise ValueError("text batch size must be positive")
    unique = sorted(set(prompts))
    if not unique:
        raise ValueError("semantic supervision has no answerable prompts")
    device = torch.device("cpu")
    determinism = configure_determinism(device, seed)
    checkpoint_before = {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path.resolve()),
        "size_bytes": checkpoint_path.resolve().stat().st_size,
    }
    source_before = source_tree_identity(repository_root.resolve())
    encoder, state_provenance = load_frozen_encoder(
        repository_root.resolve(), checkpoint_path.resolve(), device
    )
    encoded: dict[str, torch.Tensor] = {}
    try:
        with torch.inference_mode():
            for start in range(0, len(unique), batch_size):
                batch = unique[start : start + batch_size]
                embeddings = encoder.get_query_embed(modality="text", text=batch)
                if embeddings.shape != (len(batch), 512):
                    raise RuntimeError(
                        "unexpected semantic CLAP shape: "
                        f"{tuple(embeddings.shape)}"
                    )
                embeddings = embeddings.detach().float().cpu().contiguous()
                if not bool(torch.isfinite(embeddings).all()):
                    raise RuntimeError("semantic CLAP targets contain NaN or Inf")
                norms = torch.linalg.vector_norm(embeddings, dim=-1)
                if not torch.allclose(
                    norms,
                    torch.ones_like(norms),
                    rtol=1e-4,
                    atol=1e-4,
                ):
                    raise RuntimeError("semantic CLAP targets are not normalized")
                encoded.update(dict(zip(batch, embeddings.unbind(0))))
    finally:
        del encoder
        gc.collect()
    checkpoint_after = {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path.resolve()),
        "size_bytes": checkpoint_path.resolve().stat().st_size,
    }
    source_after = source_tree_identity(repository_root.resolve())
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("AudioSep checkpoint changed during semantic caching")
    if source_before != source_after:
        raise RuntimeError("AudioSep source tree changed during semantic caching")
    return encoded, {
        "batch_size": batch_size,
        "unique_prompt_count": len(unique),
        "seed": seed,
        "device": "cpu",
        "determinism": determinism,
        "query_encoder_state": state_provenance,
        "audiosep_source_tree": source_before,
        "bounded_memory": True,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_cache_atomically(
    output: Path, payload: dict[str, object], *, overwrite: bool
) -> None:
    """Publish one complete tensor cache while retaining the old file on error."""

    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    if output.exists() and not output.is_file():
        raise ValueError(f"semantic cache target is not a regular file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.building-", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        # Validate the exact bytes before publication. weights_only also keeps
        # this cache format free of arbitrary pickle objects.
        reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if not isinstance(reloaded, dict) or reloaded.get("format") != payload.get(
            "format"
        ):
            raise RuntimeError("temporary semantic cache failed reload validation")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def oracle_training_prompts(
    records: list[QCESRecord | QCESV4Record],
) -> dict[str, str]:
    """Build training-only oracle prompts identically for QCES v3 and v4.

    ``oracle_prompt`` composes answerable prompts from the labels attached to
    ``evidence_event_ids`` (the anchor and answer roles), rather than copying
    the answer string.  No-evidence records use their annotated absent label,
    matching the existing v3 supervision exactly.  These embeddings are
    targets for composer training and are not inputs to inference.
    """

    return {record.sample_id: oracle_prompt(record) for record in records}


def role_training_prompts(
    records: list[QCESRecord | QCESV4Record | QCESV5Record],
) -> tuple[dict[str, dict[str, str]], list[str], dict[str, bool]]:
    """Return role-local prompts and metadata-only same-label targets.

    No-evidence records have no meaningful anchor or answer condition.  They
    are represented explicitly by ID instead of receiving a fabricated text
    target.  The equality target comes solely from the annotated role-event
    labels and is never needed by validation/test inference.
    """

    prompts: dict[str, dict[str, str]] = {}
    no_evidence_ids: list[str] = []
    same_semantic: dict[str, bool] = {}
    for record in records:
        if record.no_evidence:
            no_evidence_ids.append(record.sample_id)
            continue
        if len(record.anchor_event_ids) != 1 or len(record.answer_event_ids) != 1:
            raise ValueError(
                f"{record.sample_id} needs exactly one event per semantic role"
            )
        anchor = record.event_by_id(record.anchor_event_ids[0])
        answer = record.event_by_id(record.answer_event_ids[0])
        prompts[record.sample_id] = {
            "anchor": describe(anchor.label),
            "answer": describe(answer.label),
        }
        same_semantic[record.sample_id] = anchor.label == answer.label
    return prompts, sorted(no_evidence_ids), same_semantic


def v5_union_training_prompts(
    records: list[QCESV5Record],
) -> tuple[dict[str, str], list[str]]:
    """Build one strict answerable-only union prompt per QCES-v5 record.

    Role events are restored to timeline order and their canonical labels are
    deduplicated before text normalization.  Consequently repeated same-label
    anchor/answer roles produce one class prompt, not ``"bell and bell"``.
    No-evidence records are an explicit disjoint ID partition and receive no
    fabricated absent-label embedding.
    """

    prompts: dict[str, str] = {}
    no_evidence_ids: list[str] = []
    for record in records:
        if record.no_evidence:
            no_evidence_ids.append(record.sample_id)
            continue
        if len(record.anchor_event_ids) != 1 or len(record.answer_event_ids) != 1:
            raise ValueError(
                f"{record.sample_id} needs exactly one event per semantic role"
            )
        role_events = sorted(
            (
                record.event_by_id(event_id)
                for event_id in (
                    *record.anchor_event_ids,
                    *record.answer_event_ids,
                )
            ),
            key=lambda event: (event.onset_seconds, event.event_id),
        )
        unique_labels: list[str] = []
        for event in role_events:
            if event.label not in unique_labels:
                unique_labels.append(event.label)
        if not unique_labels:
            raise ValueError(f"{record.sample_id} has no answerable role labels")
        prompts[record.sample_id] = " and ".join(
            describe(label) for label in unique_labels
        )
    return prompts, sorted(no_evidence_ids)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--semantic-separation-mode",
        choices=SEMANTIC_SEPARATION_MODES,
        default=UNION_SINGLE_SEMANTIC_MODE,
        help=(
            "union_single writes the historical union-target cache; dual_role "
            "writes separate anchor/answer CLAP targets and same-label metadata"
        ),
    )
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output}; use --overwrite")
    dataset = QCESManifestDataset(args.manifest.resolve(), crop_samples=None)
    if not all(
        isinstance(record, SUPPORTED_RECORD_TYPES) for record in dataset.records
    ):
        raise SystemExit("semantic target caching requires a QCES v3/v4/v5 manifest")
    records = [
        record
        for record in dataset.records
        if isinstance(record, SUPPORTED_RECORD_TYPES)
    ]
    checkpoint = args.audiosep_checkpoint.resolve()
    checkpoint_identity = {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
    }
    manifest_bytes = args.manifest.resolve().read_bytes()
    common = {
        "schema_version": records[0].schema_version,
        "target_scope": "training_supervision_only",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "audiosep_checkpoint": str(checkpoint),
        "audiosep_checkpoint_sha256": checkpoint_identity["sha256"],
        "audiosep_checkpoint_identity": checkpoint_identity,
    }
    if args.semantic_separation_mode == UNION_SINGLE_SEMANTIC_MODE:
        if all(isinstance(record, (QCESRecord, QCESV4Record)) for record in records):
            legacy_records = [
                record
                for record in records
                if isinstance(record, (QCESRecord, QCESV4Record))
            ]
            prompts = oracle_training_prompts(legacy_records)
            encoded, encoder_provenance = encode_prompts_batched(
                args.audiosep_root.resolve(),
                checkpoint,
                prompts.values(),
                batch_size=args.text_batch_size,
                seed=args.seed,
            )
            targets = {
                sample_id: encoded[prompt]
                for sample_id, prompt in prompts.items()
            }
            # Do not change this legacy payload: v3/v4 caches include their
            # historical absent-label target for no-evidence records.
            payload = {
                **common,
                "format": "qces_audiosep_semantic_targets_v1",
                "prompt_source": "evidence_role_event_labels_or_absent_label",
                "prompts": prompts,
                "targets": targets,
            }
        elif all(isinstance(record, QCESV5Record) for record in records):
            v5_records = [
                record for record in records if isinstance(record, QCESV5Record)
            ]
            prompts, no_evidence_ids = v5_union_training_prompts(v5_records)
            encoded, encoder_provenance = encode_prompts_batched(
                args.audiosep_root.resolve(),
                checkpoint,
                prompts.values(),
                batch_size=args.text_batch_size,
                seed=args.seed,
            )
            targets = {
                sample_id: encoded[prompt]
                for sample_id, prompt in prompts.items()
            }
            payload = {
                **common,
                "format": V5_UNION_SEMANTIC_CACHE_FORMAT,
                "target_scope": V5_UNION_SEMANTIC_TARGET_SCOPE,
                "prompt_source": V5_UNION_SEMANTIC_PROMPT_SOURCE,
                "prompts": prompts,
                "targets": targets,
                "no_evidence_ids": no_evidence_ids,
            }
        else:
            raise SystemExit(
                "union semantic cache requires one pure QCES v3, v4, or v5 "
                "schema"
            )
    elif args.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE:
        prompts, no_evidence_ids, same_semantic = role_training_prompts(records)
        unique_prompts = [
            prompt
            for role_prompts in prompts.values()
            for prompt in role_prompts.values()
        ]
        encoded, encoder_provenance = encode_prompts_batched(
            args.audiosep_root.resolve(),
            checkpoint,
            unique_prompts,
            batch_size=args.text_batch_size,
            seed=args.seed,
        )
        role_targets = {
            sample_id: {
                "anchor": encoded[role_prompts["anchor"]],
                "answer": encoded[role_prompts["answer"]],
                "same_semantic": same_semantic[sample_id],
            }
            for sample_id, role_prompts in prompts.items()
        }
        payload = {
            **common,
            "format": ROLE_SEMANTIC_CACHE_FORMAT,
            "target_scope": ROLE_SEMANTIC_TARGET_SCOPE,
            "prompt_source": ROLE_SEMANTIC_PROMPT_SOURCE,
            "role_prompts": prompts,
            "role_targets": role_targets,
            "no_evidence_ids": no_evidence_ids,
        }
        targets = role_targets
    else:  # argparse validates choices.
        raise RuntimeError("unsupported semantic separation mode")
    payload["text_encoder_provenance"] = encoder_provenance
    write_cache_atomically(output, payload, overwrite=args.overwrite)
    print(
        f"cached {len(targets)} AudioSep semantic records "
        f"({args.semantic_separation_mode}) at {output}"
    )


if __name__ == "__main__":
    main()
