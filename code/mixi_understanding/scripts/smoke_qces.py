#!/usr/bin/env python3
"""Run one end-to-end QCES optimization step without data or checkpoints."""
# flake8: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.qces.config import QCESConfig
from mixi_understanding.qces.losses import QCESLoss
from mixi_understanding.qces.model import QCESModel
from mixi_understanding.qces.tokenization import StableHashTokenizer


def main() -> None:
    torch.manual_seed(7)
    config = QCESConfig(
        sample_rate=8_000,
        n_fft=128,
        hop_length=32,
        win_length=128,
        vocab_size=256,
        max_question_tokens=16,
        question_dim=32,
        audio_dim=32,
        condition_dim=32,
        attention_heads=4,
        question_layers=1,
        separator_channels=8,
        separator_layers=2,
        dropout=0.0,
    )
    samples = 2_048
    time = torch.arange(samples) / config.sample_rate
    anchor = torch.sin(2 * torch.pi * 330 * time) * (time < 0.08)
    answer = torch.sin(2 * torch.pi * 660 * time) * (
        (time >= 0.12) & (time < 0.20)
    )
    nuisance = 0.15 * torch.randn_like(time)
    evidence = (anchor + answer)[None]
    residual = nuisance[None]
    mixture = evidence + residual
    anchor_mask = ((time < 0.08)).float()[None]
    answer_mask = (((time >= 0.12) & (time < 0.20))).float()[None]
    tokenizer = StableHashTokenizer(config.vocab_size, config.max_question_tokens)
    tokens = tokenizer.batch_encode(["What sound occurs after the first tone?"])
    batch = {
        "mixture": mixture,
        "evidence": evidence,
        "residual": residual,
        "anchor_mask": anchor_mask,
        "answer_mask": answer_mask,
        "no_evidence": torch.zeros(1),
        "question_ids": tokens.input_ids,
        "question_mask": tokens.attention_mask,
    }
    model = QCESModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output = model(mixture, tokens.input_ids, tokens.attention_mask)
    loss, components = QCESLoss(fft_sizes=(64, 128))(output, batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("smoke loss is not finite")
    if output.separation.mixture_error.max() > 1e-5:
        raise RuntimeError("evidence and residual do not reconstruct the mixture")
    print(
        "QCES smoke test passed: "
        f"loss={float(loss):.6f}, "
        f"mixture_error={float(output.separation.mixture_error.max()):.3e}, "
        f"terms={len(components)}"
    )


if __name__ == "__main__":
    main()
