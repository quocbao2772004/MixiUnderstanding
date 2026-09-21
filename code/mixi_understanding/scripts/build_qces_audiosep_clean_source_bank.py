#!/usr/bin/env python3
"""Clean requested AudioSet event crops with frozen official AudioSep.

This command is intentionally a sidecar to the historical QCES pipelines.  It
uses the exact official AudioSep checkpoint/config plus the checkpoint's CLAP
query encoder, deterministically normalizes each native 44.1/48 kHz AudioSet
crop to AudioSep's canonical 32 kHz input, and never reads a QA question,
answer, or downstream metric.  The resampler protocol and both native/input
hashes are committed in every source-bank fragment and receipt.
Use ``--max-items`` for a bounded smoke run; this script does not launch a full
job on its own.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from mixi_understanding.qces.audiosep_source_cleaner import (
    SourceBankConfig,
    build_source_bank,
    sha256_file,
)
from mixi_understanding.qces.separators import AudioSepConditionedAdapter
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    file_identity,
    source_tree_identity,
)
from mixi_understanding.scripts.cache_qces_v6_stem_features import (
    load_clap_encoder,
)


DEFAULT_AUDIOSEP_ROOT = PROJECT_ROOT / "code/baseline/audiosep"
DEFAULT_AUDIOSEP_CONFIG = DEFAULT_AUDIOSEP_ROOT / "config/audiosep_base.yaml"
DEFAULT_AUDIOSEP_CHECKPOINT = (
    DEFAULT_AUDIOSEP_ROOT / "checkpoint/hf_audiosep/pytorch_model.bin"
)
CANONICAL_SAMPLE_RATE = 32_000
CLAP_CANONICAL_SAMPLES = 320_000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Requested-crop manifest; repeat for multiple materialization outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, default=DEFAULT_AUDIOSEP_ROOT)
    parser.add_argument("--audiosep-config", type=Path, default=DEFAULT_AUDIOSEP_CONFIG)
    parser.add_argument(
        "--audiosep-checkpoint", type=Path, default=DEFAULT_AUDIOSEP_CHECKPOINT
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    # The core submits primary+paraphrase together, so batch-size=2 is four
    # simultaneous 10-second separator queries and fits the project T4 safely.
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Deterministic tier-ordered smoke cap; 0 processes the complete input.",
    )
    parser.add_argument("--target-train-per-class", type=int, default=100)
    parser.add_argument("--target-eval-per-class", type=int, default=20)
    parser.add_argument(
        "--store-residual",
        action="store_true",
        help=(
            "Persist source-crop residual FLAC files. The residual always participates "
            "in quality scoring; by default only the downstream clean stem is retained."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


class OfficialAudioSepCleanerBackend:
    """Batched frozen AudioSep separation and frozen AudioSep-CLAP scoring."""

    sample_rate = CANONICAL_SAMPLE_RATE

    def __init__(
        self,
        *,
        repository_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        device: torch.device,
        embedding_batch_size: int,
    ) -> None:
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        self.repository_root = repository_root.resolve()
        self.config_path = config_path.resolve()
        self.checkpoint_path = checkpoint_path.resolve()
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        self.separator = AudioSepConditionedAdapter.from_repository(
            repository_root=self.repository_root,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            device=device,
            freeze_separator=True,
        ).ss_model.eval()
        self.encoder = load_clap_encoder(
            self.repository_root, self.checkpoint_path, device
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        self._text_cache: dict[str, torch.Tensor] = {}
        self._identity = {
            "backend": "official_frozen_audiosep_resunet30_and_checkpoint_clap",
            "audiosep_checkpoint": file_identity(self.checkpoint_path),
            "audiosep_config": file_identity(self.config_path),
            "audiosep_source_tree": source_tree_identity(self.repository_root),
            "sample_rate": self.sample_rate,
            "separator_frozen": True,
            "clap_frozen": True,
            "primary_prompt_policy": "exact_official_display_name",
            "paraphrase_policy": "the sound of {exact_official_display_name}",
            "clap_audio_length_policy": "repeatpad_to_exact_10_seconds_before_encoder",
        }

    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def _text_tensors(self, prompts: Sequence[str]) -> torch.Tensor:
        missing = list(dict.fromkeys(prompt for prompt in prompts if prompt not in self._text_cache))
        with torch.inference_mode():
            for start in range(0, len(missing), self.embedding_batch_size):
                batch = missing[start : start + self.embedding_batch_size]
                embeddings = self.encoder.get_query_embed(modality="text", text=batch)
                for prompt, vector in zip(batch, embeddings):
                    self._text_cache[prompt] = vector.detach().cpu()
        return torch.stack([self._text_cache[prompt] for prompt in prompts]).to(
            self.device
        )

    @staticmethod
    def _pad_separator_batch(waveforms: Sequence[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
        lengths = [int(np.asarray(value).size) for value in waveforms]
        if not lengths or min(lengths) <= 0:
            raise ValueError("separator batch contains empty waveform")
        # ResUNet is fully convolutional; zero padding is removed after the
        # forward.  Aligning to 1024 keeps its encoder/decoder grid stable.
        padded_samples = max(4096, int(math.ceil(max(lengths) / 1024.0) * 1024))
        output = torch.zeros((len(waveforms), padded_samples), dtype=torch.float32)
        for index, waveform in enumerate(waveforms):
            tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
            output[index, : tensor.numel()] = tensor
        return output, lengths

    def separate(
        self, waveforms: Sequence[np.ndarray], prompts: Sequence[str]
    ) -> Sequence[np.ndarray]:
        if len(waveforms) != len(prompts):
            raise ValueError("waveforms and prompts must have equal length")
        mixture, lengths = self._pad_separator_batch(waveforms)
        condition = self._text_tensors(prompts)
        with torch.inference_mode():
            result = self.separator(
                {
                    "mixture": mixture[:, None].to(self.device),
                    "condition": condition,
                }
            )
            raw = result["waveform"] if isinstance(result, dict) else result
            if raw.ndim == 3 and raw.shape[1] == 1:
                raw = raw[:, 0]
            raw = raw.detach().cpu()
        if raw.shape != mixture.shape:
            raise RuntimeError(
                f"AudioSep output shape {tuple(raw.shape)} != {tuple(mixture.shape)}"
            )
        return [
            raw[index, :length].numpy().astype(np.float32, copy=True)
            for index, length in enumerate(lengths)
        ]

    @staticmethod
    def _canonical_clap_audio(waveform: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        if tensor.numel() <= 0:
            raise ValueError("cannot embed empty audio")
        if tensor.numel() < CLAP_CANONICAL_SAMPLES:
            repeat = CLAP_CANONICAL_SAMPLES // tensor.numel()
            tensor = tensor.repeat(repeat)
            tensor = F.pad(tensor, (0, CLAP_CANONICAL_SAMPLES - tensor.numel()))
        else:
            # Event crops are at most the 10-second AudioSet segment.  Explicit
            # front truncation is only a fail-safe and avoids CLAP's stochastic
            # fusion truncation branch.
            tensor = tensor[:CLAP_CANONICAL_SAMPLES]
        if tensor.numel() != CLAP_CANONICAL_SAMPLES:
            raise RuntimeError("CLAP canonicalization produced a wrong length")
        return tensor

    def embed_audio(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        output: list[torch.Tensor] = []
        canonical = [self._canonical_clap_audio(value) for value in waveforms]
        with torch.inference_mode():
            for start in range(0, len(canonical), self.embedding_batch_size):
                batch = torch.stack(
                    canonical[start : start + self.embedding_batch_size]
                ).to(self.device)
                output.append(
                    self.encoder.get_query_embed(modality="audio", audio=batch)
                    .detach()
                    .cpu()
                )
        return torch.cat(output, dim=0).numpy()

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        return self._text_tensors(texts).detach().cpu().numpy()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1 or args.embedding_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    backend = OfficialAudioSepCleanerBackend(
        repository_root=args.audiosep_root,
        config_path=args.audiosep_config,
        checkpoint_path=args.audiosep_checkpoint,
        device=device,
        embedding_batch_size=args.embedding_batch_size,
    )
    receipt = build_source_bank(
        SourceBankConfig(
            manifest_paths=tuple(args.manifest),
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            max_items=args.max_items,
            target_train_per_class=args.target_train_per_class,
            target_eval_per_class=args.target_eval_per_class,
            store_residual=args.store_residual,
        ),
        backend,
    )
    print(
        f"wrote={args.output_dir.resolve()} input={receipt['input_items']} "
        f"accepted={receipt['accepted_items']} rejected={receipt['rejected_items']} "
        f"resumed={receipt['resumed_items']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
