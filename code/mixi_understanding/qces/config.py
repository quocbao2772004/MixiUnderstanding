"""Configuration for question-conditioned evidence separation (QCES)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


SEPARATOR_AWARE_REFINER_MODES = ("full", "relative_energy")
LEGACY_TEMPORAL_ROLE_MODE = "exclusive_softmax"
OVERLAP_AWARE_TEMPORAL_ROLE_MODE = "independent_sigmoid"
TEMPORAL_ROLE_MODES = (
    LEGACY_TEMPORAL_ROLE_MODE,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
)
UNION_SINGLE_SEMANTIC_MODE = "union_single"
DUAL_ROLE_SEMANTIC_MODE = "dual_role"
SEMANTIC_SEPARATION_MODES = (
    UNION_SINGLE_SEMANTIC_MODE,
    DUAL_ROLE_SEMANTIC_MODE,
)
NO_FOUNDATION_FEATURES = "none"
AUDIOSEP_CLAP_FOUNDATION_FEATURES = "audiosep_clap"
FOUNDATION_FEATURE_MODES = (
    NO_FOUNDATION_FEATURES,
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
)
LEGACY_BOUNDED_SEMANTIC_RESIDUAL = "bounded_residual"
CONVEX_SEMANTIC_INTERPOLATION = "convex_interpolation"
QUESTION_RESIDUAL_SEMANTIC_MIXING = "question_residual"
FOUNDATION_SEMANTIC_MIXING_MODES = (
    LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
    CONVEX_SEMANTIC_INTERPOLATION,
    QUESTION_RESIDUAL_SEMANTIC_MIXING,
)


@dataclass(frozen=True)
class QCESConfig:
    """Shape and signal-processing parameters shared by the QCES modules."""

    sample_rate: int = 32_000
    n_fft: int = 512
    hop_length: int = 320
    win_length: int = 512
    vocab_size: int = 8_192
    max_question_tokens: int = 48
    question_dim: int = 128
    audio_dim: int = 128
    condition_dim: int = 512
    attention_heads: int = 4
    question_layers: int = 2
    separator_channels: int = 32
    separator_layers: int = 4
    dropout: float = 0.1
    # Keep the historical mutually-exclusive head as the default so old
    # checkpoints that predate this field load with exactly their old
    # semantics. Paper-scale QCES-v5 runs must opt into independent_sigmoid.
    temporal_role_mode: str = LEGACY_TEMPORAL_ROLE_MODE
    # ``union_single`` is the historical one-condition/one-separator-evaluation
    # path.  ``dual_role`` is an explicit candidate that predicts separate
    # anchor/answer conditions and therefore changes compute and checkpoint
    # semantics.  Keeping the legacy mode as the default makes old checkpoint
    # payloads load without adding trainable parameters to their composer.
    semantic_separation_mode: str = UNION_SINGLE_SEMANTIC_MODE
    separator_aware_refiner: bool = False
    separator_aware_refiner_mode: str = "full"
    # Offline frozen-CLAP inputs are an opt-in training candidate. Keeping
    # ``none`` as the default creates no extra modules, parameters, or state
    # keys and preserves the historical forward operation sequence exactly.
    foundation_feature_mode: str = NO_FOUNDATION_FEATURES
    # The legacy residual can never make the learned candidate stronger than
    # its acoustic base. ``convex_interpolation`` reaches a learned endpoint;
    # ``question_residual`` instead starts from the official full-question CLAP
    # condition and learns an unbounded audio-question correction. Retaining
    # the legacy default makes old checkpoint payloads and operations exact.
    foundation_semantic_mixing_mode: str = LEGACY_BOUNDED_SEMANTIC_RESIDUAL

    def __post_init__(self) -> None:
        positive = {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "vocab_size": self.vocab_size,
            "max_question_tokens": self.max_question_tokens,
            "question_dim": self.question_dim,
            "audio_dim": self.audio_dim,
            "condition_dim": self.condition_dim,
            "attention_heads": self.attention_heads,
            "question_layers": self.question_layers,
            "separator_channels": self.separator_channels,
            "separator_layers": self.separator_layers,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"QCES dimensions must be positive, got {invalid}")
        if self.vocab_size < 4:
            raise ValueError("vocab_size must leave room for reserved tokens")
        if self.win_length > self.n_fft:
            raise ValueError("win_length must be <= n_fft")
        if self.question_dim % self.attention_heads != 0:
            raise ValueError("question_dim must be divisible by attention_heads")
        if self.audio_dim % self.attention_heads != 0:
            raise ValueError("audio_dim must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.separator_aware_refiner, bool):
            raise ValueError("separator_aware_refiner must be a boolean")
        if (
            not isinstance(self.temporal_role_mode, str)
            or self.temporal_role_mode not in TEMPORAL_ROLE_MODES
        ):
            raise ValueError(
                "temporal_role_mode must be one of " f"{TEMPORAL_ROLE_MODES}"
            )
        if (
            not isinstance(self.semantic_separation_mode, str)
            or self.semantic_separation_mode not in SEMANTIC_SEPARATION_MODES
        ):
            raise ValueError(
                "semantic_separation_mode must be one of "
                f"{SEMANTIC_SEPARATION_MODES}"
            )
        if (
            not isinstance(self.foundation_feature_mode, str)
            or self.foundation_feature_mode not in FOUNDATION_FEATURE_MODES
        ):
            raise ValueError(
                "foundation_feature_mode must be one of " f"{FOUNDATION_FEATURE_MODES}"
            )
        if (
            not isinstance(self.foundation_semantic_mixing_mode, str)
            or self.foundation_semantic_mixing_mode
            not in FOUNDATION_SEMANTIC_MIXING_MODES
        ):
            raise ValueError(
                "foundation_semantic_mixing_mode must be one of "
                f"{FOUNDATION_SEMANTIC_MIXING_MODES}"
            )
        if (
            self.foundation_feature_mode == NO_FOUNDATION_FEATURES
            and self.foundation_semantic_mixing_mode != LEGACY_BOUNDED_SEMANTIC_RESIDUAL
        ):
            raise ValueError(
                "a non-legacy foundation semantic mixing mode requires "
                "foundation_feature_mode='audiosep_clap'"
            )
        if (
            self.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES
            and self.condition_dim != 512
        ):
            raise ValueError(
                "audiosep_clap foundation features require condition_dim=512"
            )
        if (
            not isinstance(self.separator_aware_refiner_mode, str)
            or self.separator_aware_refiner_mode not in SEPARATOR_AWARE_REFINER_MODES
        ):
            raise ValueError(
                "separator_aware_refiner_mode must be one of "
                f"{SEPARATOR_AWARE_REFINER_MODES}"
            )
        if (
            not self.separator_aware_refiner
            and self.separator_aware_refiner_mode != "full"
        ):
            raise ValueError(
                "a non-default separator_aware_refiner_mode requires "
                "separator_aware_refiner=True"
            )
        if (
            self.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
            and self.temporal_role_mode != OVERLAP_AWARE_TEMPORAL_ROLE_MODE
        ):
            raise ValueError(
                "dual_role semantic separation requires overlap-aware "
                "independent_sigmoid temporal roles"
            )
        if (
            self.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
            and self.separator_aware_refiner
        ):
            raise ValueError(
                "dual_role semantic separation is not compatible with the "
                "single-raw-stem separator-aware refiner"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QCESConfig":
        return cls(**payload)
