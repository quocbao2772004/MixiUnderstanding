#!/usr/bin/env python3
"""Cache frozen AudioSep-CLAP text embeddings for the 191-label ontology."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.scripts.cache_audiosep_semantic_targets import (
    encode_prompts_batched,
    sha256_file,
)
from mixi_understanding.scripts.evaluate_audiosep_baselines import describe


FORMAT = "qces_label_clap191_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ontology", type=Path,
        default=PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/ontology_191.txt",
    )
    parser.add_argument(
        "--audiosep-root", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep",
    )
    parser.add_argument(
        "--audiosep-checkpoint", type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/qces_label_clap191_v1.pt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2103)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite label cache: {output}")
    labels = [
        line.strip() for line in args.ontology.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != 191 or len(labels) != len(set(labels)):
        raise ValueError(f"requires frozen 191-label ontology, got {len(labels)}")
    prompts = [f"the sound of {describe(label).replace('_', ' ').strip()}" for label in labels]
    encoded, provenance = encode_prompts_batched(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(), prompts,
        batch_size=args.batch_size, seed=args.seed,
    )
    embeddings = torch.stack([encoded[prompt] for prompt in prompts]).float().contiguous()
    if embeddings.shape != (191, 512) or not bool(torch.isfinite(embeddings).all()):
        raise RuntimeError(f"invalid CLAP label embedding matrix: {tuple(embeddings.shape)}")
    payload = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "prompts": prompts,
        "embeddings": embeddings,
        "ontology_sha256": sha256_file(args.ontology.resolve()),
        "audiosep_checkpoint_sha256": sha256_file(args.audiosep_checkpoint.resolve()),
        "provenance": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        check = torch.load(temporary, map_location="cpu", weights_only=True)
        if check.get("format") != FORMAT or tuple(check["embeddings"].shape) != (191, 512):
            raise RuntimeError("temporary CLAP label cache failed validation")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "format": FORMAT,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "labels": len(labels),
        "embedding_shape": list(embeddings.shape),
        "prompt_template": "the sound of {humanized_label}",
    }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
