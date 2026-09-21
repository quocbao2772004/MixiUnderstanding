"""Dependency-free, deterministic question tokenization for QCES baselines."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


PAD_TOKEN_ID = 0
EMPTY_TOKEN_ID = 1
_TOKEN_PATTERN = re.compile(r"[\w']+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "TokenBatch":
        return TokenBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )


class StableHashTokenizer:
    """Map Unicode tokens to stable hash buckets without a fitted vocabulary.

    This tokenizer keeps the first runnable implementation self-contained. Paper
    experiments can replace it with frozen Audio Flamingo/T5 text features while
    preserving the rest of the model interface.
    """

    def __init__(self, vocab_size: int = 8_192, max_length: int = 48) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.vocab_size = vocab_size
        self.max_length = max_length

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError("question must be a string")
        return _TOKEN_PATTERN.findall(text.casefold())[: self.max_length]

    def token_to_id(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket_count = self.vocab_size - 2
        return 2 + int.from_bytes(digest[:8], "big") % bucket_count

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        if not tokens:
            return [EMPTY_TOKEN_ID]
        return [self.token_to_id(token) for token in tokens]

    def batch_encode(
        self,
        questions: Sequence[str],
        device: torch.device | str | None = None,
    ) -> TokenBatch:
        if not questions:
            raise ValueError("questions must not be empty")
        rows = [self.encode(question) for question in questions]
        width = min(self.max_length, max(len(row) for row in rows))
        input_ids = torch.full(
            (len(rows), width), PAD_TOKEN_ID, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (len(rows), width), dtype=torch.bool, device=device
        )
        for index, row in enumerate(rows):
            clipped = row[:width]
            input_ids[index, : len(clipped)] = torch.tensor(
                clipped, dtype=torch.long, device=device
            )
            attention_mask[index, : len(clipped)] = True
        return TokenBatch(input_ids=input_ids, attention_mask=attention_mask)

    def iter_token_ids(self, texts: Iterable[str]) -> Iterable[List[int]]:
        for text in texts:
            yield self.encode(text)
