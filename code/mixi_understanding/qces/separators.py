"""Separator backends for question-conditioned acoustic evidence rendering."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mixi_understanding.qces.composer import (
    ROLE_ANCHOR,
    ROLE_ANSWER,
    PromptComposition,
    temporal_evidence_probability,
)
from mixi_understanding.qces.config import (
    DUAL_ROLE_SEMANTIC_MODE,
    QCESConfig,
    UNION_SINGLE_SEMANTIC_MODE,
)
from mixi_understanding.qces.signal import (
    deterministic_reflect_pad_1d,
    qces_linear_interpolate_1d,
    qces_smooth_1d,
    qces_stft,
)


@dataclass
class SeparationOutput:
    evidence: torch.Tensor
    residual: torch.Tensor
    mask: Optional[torch.Tensor]
    raw_mask: Optional[torch.Tensor]
    raw_evidence: Optional[torch.Tensor]
    mixture_error: torch.Tensor
    refined_composition: Optional[PromptComposition] = None
    semantic_separation_mode: str = UNION_SINGLE_SEMANTIC_MODE
    physical_separator_forwards_per_batch: int = 1
    effective_separator_evaluations_per_record: int = 1
    anchor_raw_evidence: Optional[torch.Tensor] = None
    answer_raw_evidence: Optional[torch.Tensor] = None
    same_semantic_probability: Optional[torch.Tensor] = None


def _mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 3 and waveform.size(1) == 1:
        waveform = waveform[:, 0]
    if waveform.ndim != 2:
        raise ValueError("mixture must have shape [B, N] or [B, 1, N]")
    return waveform


def _deterministic_torchlibrosa_stft_forward(
    stft: nn.Module, input_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torchlibrosa STFT forward with an exact deterministic reflect pad."""

    x = input_tensor[:, None, :]
    if stft.center:
        padding = int(stft.n_fft) // 2
        if stft.pad_mode == "reflect":
            x = deterministic_reflect_pad_1d(x, padding, padding)
        else:
            x = F.pad(x, pad=(padding, padding), mode=stft.pad_mode)
    real = stft.conv_real(x)
    imaginary = stft.conv_imag(x)
    return (
        real[:, None, :, :].transpose(2, 3),
        imaginary[:, None, :, :].transpose(2, 3),
    )


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, 2 * channels)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = self.norm(features)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = F.gelu(self.conv1(hidden))
        return features + self.conv2(hidden)


class ComplementaryMaskSeparator(nn.Module):
    """Render evidence/residual with complementary real-valued STFT masks.

    This is the checkpoint-free default. Both stems share the input phase and
    complementary masks, which prevents generative event insertion and gives
    numerical mixture consistency by construction.
    """

    def __init__(self, config: QCESConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "window", torch.hann_window(config.win_length), persistent=False
        )
        channels = config.separator_channels
        self.input = nn.Conv2d(1, channels, 5, padding=2)
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(channels, config.condition_dim)
            for _ in range(config.separator_layers)
        )
        groups = 8 if channels % 8 == 0 else 1
        self.output_norm = nn.GroupNorm(groups, channels)
        self.output = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, mixture: torch.Tensor, composition: PromptComposition
    ) -> SeparationOutput:
        mixture = _mono_waveform(mixture)
        spectrum = qces_stft(
            mixture,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(device=mixture.device, dtype=mixture.dtype),
            center=True,
            return_complex=True,
        )
        log_magnitude = torch.log1p(spectrum.abs())
        scale = log_magnitude.square().mean(dim=(1, 2), keepdim=True).sqrt()
        features = self.input((log_magnitude / scale.clamp_min(1e-5)).unsqueeze(1))
        for block in self.blocks:
            features = block(features, composition.semantic_condition)
        normalized = F.gelu(self.output_norm(features))
        raw_mask = torch.sigmoid(self.output(normalized).squeeze(1))

        temporal_gate = qces_linear_interpolate_1d(
            composition.evidence_probability[:, None], spectrum.size(-1)
        ).squeeze(1)
        # The temporal role head already represents whether evidence is present.
        # Multiplying by a second no-evidence gate creates an easy all-silent
        # shortcut and weakens gradients to the separator.
        mask = raw_mask * temporal_gate[:, None]
        evidence_spectrum = spectrum * mask
        residual_spectrum = spectrum * (1.0 - mask)
        window = self.window.to(device=mixture.device, dtype=mixture.dtype)
        evidence = torch.istft(
            evidence_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=True,
            length=mixture.size(-1),
        )
        residual = torch.istft(
            residual_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=True,
            length=mixture.size(-1),
        )
        mixture_error = (mixture - evidence - residual).abs().mean(dim=-1)
        return SeparationOutput(
            evidence=evidence,
            residual=residual,
            mask=mask,
            raw_mask=raw_mask,
            raw_evidence=None,
            mixture_error=mixture_error,
        )


class PhaseAwareComplexMaskSeparator(nn.Module):
    """Render complementary stems with a bounded complex ratio mask."""

    def __init__(self, config: QCESConfig, max_mask_magnitude: float = 2.0) -> None:
        super().__init__()
        if max_mask_magnitude <= 0:
            raise ValueError("max_mask_magnitude must be positive")
        self.config = config
        self.max_mask_magnitude = max_mask_magnitude
        self.register_buffer(
            "window", torch.hann_window(config.win_length), persistent=False
        )
        channels = config.separator_channels
        self.input = nn.Conv2d(3, channels, 5, padding=2)
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(channels, config.condition_dim)
            for _ in range(config.separator_layers)
        )
        groups = 8 if channels % 8 == 0 else 1
        self.output_norm = nn.GroupNorm(groups, channels)
        self.output = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        initial_real = max_mask_magnitude * torch.atanh(
            torch.tensor(0.5 / max_mask_magnitude)
        )
        with torch.no_grad():
            self.output.bias[0] = initial_real

    def _bound_mask(self, logits: torch.Tensor) -> torch.Tensor:
        raw = torch.complex(logits[:, 0], logits[:, 1])
        magnitude = raw.abs()
        bounded_magnitude = self.max_mask_magnitude * torch.tanh(
            magnitude / self.max_mask_magnitude
        )
        return raw * (bounded_magnitude / magnitude.clamp_min(1e-8))

    def forward(
        self, mixture: torch.Tensor, composition: PromptComposition
    ) -> SeparationOutput:
        mixture = _mono_waveform(mixture)
        window = self.window.to(device=mixture.device, dtype=mixture.dtype)
        spectrum = qces_stft(
            mixture,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        log_magnitude = torch.log1p(magnitude)
        scale = log_magnitude.square().mean(dim=(1, 2), keepdim=True).sqrt()
        phase = spectrum / magnitude.clamp_min(1e-8)
        separator_input = torch.stack(
            [
                log_magnitude / scale.clamp_min(1e-5),
                phase.real,
                phase.imag,
            ],
            dim=1,
        )
        features = self.input(separator_input)
        for block in self.blocks:
            features = block(features, composition.semantic_condition)
        normalized = F.gelu(self.output_norm(features))
        raw_mask = self._bound_mask(self.output(normalized))
        temporal_gate = qces_linear_interpolate_1d(
            composition.evidence_probability[:, None], spectrum.size(-1)
        ).squeeze(1)
        mask = raw_mask * temporal_gate[:, None]
        evidence_spectrum = spectrum * mask
        evidence = torch.istft(
            evidence_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=True,
            length=mixture.size(-1),
        )
        residual = mixture - evidence
        mixture_error = (mixture - evidence - residual).abs().mean(dim=-1)
        return SeparationOutput(
            evidence=evidence,
            residual=residual,
            mask=mask,
            raw_mask=raw_mask,
            raw_evidence=None,
            mixture_error=mixture_error,
        )


class SeparatorAwareTemporalRefiner(nn.Module):
    """Correct temporal roles after observing AudioSep's raw output.

    AudioSep is still called exactly once. ``full`` sees raw-stem spectrogram
    features, relative energy, and base composer frames. The low-capacity
    ``relative_energy`` ablation sees only the raw-stem/mixture frame-energy
    ratio and the base none/anchor/answer logits. Both predict *residual* role
    logits and a residual no-evidence logit. Their output heads are zero
    initialized, so enabling either mode preserves the base composer's gate
    exactly before the first optimizer update. The three residual logits retain
    their checkpointed none/anchor/answer identities under both legacy softmax
    and paper-profile independent sigmoid; the latter is rendered with the same
    probabilistic anchor/answer union as the base composer.
    """

    def __init__(self, config: QCESConfig) -> None:
        super().__init__()
        self.config = config
        self.mode = config.separator_aware_refiner_mode
        if self.mode == "full":
            self.register_buffer(
                "window", torch.hann_window(config.win_length), persistent=False
            )
            frequency_bins = config.n_fft // 2 + 1
            groups = 8 if config.audio_dim % 8 == 0 else 1
            # Keep the original full-mode module names and shapes unchanged so
            # pre-mode checkpoints load bit-for-bit as mode="full".
            self.stem_encoder = nn.Sequential(
                nn.Conv1d(frequency_bins, config.audio_dim, kernel_size=5, padding=2),
                nn.GroupNorm(groups, config.audio_dim),
                nn.GELU(),
                nn.Conv1d(
                    config.audio_dim,
                    config.audio_dim,
                    kernel_size=5,
                    padding=2,
                    groups=config.audio_dim,
                ),
                nn.Conv1d(config.audio_dim, config.audio_dim, kernel_size=1),
                nn.GELU(),
            )
            self.fusion = nn.Sequential(
                nn.Conv1d(
                    2 * config.audio_dim + 1,
                    config.audio_dim,
                    kernel_size=5,
                    padding=2,
                ),
                nn.GroupNorm(groups, config.audio_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Conv1d(
                    config.audio_dim,
                    config.audio_dim,
                    kernel_size=5,
                    padding=2,
                ),
                nn.GELU(),
            )
            hidden_channels = config.audio_dim
        elif self.mode == "relative_energy":
            # This ablation has no STFT/spectrogram encoder. Its only learned
            # input is one raw-stem/mixture log-RMS ratio alongside the three
            # base role logits. Residual logits preserve anchor/answer meaning.
            hidden_channels = 16
            self.energy_fusion = nn.Sequential(
                nn.Conv1d(
                    1 + 3,
                    hidden_channels,
                    kernel_size=5,
                    padding=2,
                ),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        else:  # QCESConfig validates this, but fail closed for direct mutation.
            raise ValueError(f"unsupported separator-aware refiner mode: {self.mode}")
        self.role_delta_head = nn.Conv1d(hidden_channels, 3, kernel_size=1)
        self.no_evidence_delta_head = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.role_delta_head.weight)
        nn.init.zeros_(self.role_delta_head.bias)
        nn.init.zeros_(self.no_evidence_delta_head.weight)
        nn.init.zeros_(self.no_evidence_delta_head.bias)

    def _separator_features(
        self, mixture: torch.Tensor, raw_evidence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.window.to(device=mixture.device, dtype=mixture.dtype)
        # One batched STFT keeps the two energy measurements on an identical
        # deterministic frame grid.
        spectrum = qces_stft(
            torch.cat([mixture, raw_evidence], dim=0),
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        mixture_spectrum, stem_spectrum = spectrum.chunk(2, dim=0)
        stem_log_magnitude = torch.log1p(stem_spectrum.abs())
        scale = stem_log_magnitude.square().mean(dim=(1, 2), keepdim=True).sqrt()
        stem_features = self.stem_encoder(stem_log_magnitude / scale.clamp_min(1e-5))
        mixture_energy = mixture_spectrum.abs().square().mean(dim=1)
        stem_energy = stem_spectrum.abs().square().mean(dim=1)
        relative_log_rms = 0.5 * (
            torch.log(stem_energy.clamp_min(1e-8))
            - torch.log(mixture_energy.clamp_min(1e-8))
        )
        # Bounding prevents a nearly silent mixture frame from dominating the
        # trainable fusion while retaining roughly 104 dB of dynamic range.
        relative_log_rms = relative_log_rms.clamp(-12.0, 12.0)
        return stem_features, relative_log_rms[:, None]

    @staticmethod
    def _relative_waveform_log_rms(
        mixture: torch.Tensor,
        raw_evidence: torch.Tensor,
        frames: int,
    ) -> torch.Tensor:
        """Frame raw/mixture RMS ratio without constructing a spectrogram."""

        if frames <= 0:
            raise ValueError("relative-energy refiner needs at least one frame")
        paired = torch.stack([mixture, raw_evidence], dim=1).square()
        mean_square = F.adaptive_avg_pool1d(paired, frames)
        mixture_energy = mean_square[:, 0]
        stem_energy = mean_square[:, 1]
        relative_log_rms = 0.5 * (
            torch.log(stem_energy.clamp_min(1e-8))
            - torch.log(mixture_energy.clamp_min(1e-8))
        )
        return relative_log_rms.clamp(-12.0, 12.0)[:, None]

    def forward(
        self,
        mixture: torch.Tensor,
        raw_evidence: torch.Tensor,
        base: PromptComposition,
    ) -> PromptComposition:
        mixture = _mono_waveform(mixture)
        raw_evidence = _mono_waveform(raw_evidence)
        if mixture.shape != raw_evidence.shape:
            raise ValueError("mixture and raw AudioSep evidence must have equal shape")
        target_frames = base.role_logits.size(1)
        if self.mode == "full":
            base_frames = base.frame_features.transpose(1, 2)
            stem_features, relative_energy = self._separator_features(
                mixture, raw_evidence
            )
            if stem_features.size(-1) != target_frames:
                stem_features = qces_linear_interpolate_1d(stem_features, target_frames)
                relative_energy = qces_linear_interpolate_1d(
                    relative_energy, target_frames
                )
            fused = self.fusion(
                torch.cat([base_frames, stem_features, relative_energy], dim=1)
            )
        else:
            relative_energy = self._relative_waveform_log_rms(
                mixture, raw_evidence, target_frames
            )
            fused = self.energy_fusion(
                torch.cat([base.role_logits.transpose(1, 2), relative_energy], dim=1)
            )
        role_delta = qces_smooth_1d(self.role_delta_head(fused)).transpose(1, 2)
        role_logits = base.role_logits + role_delta
        evidence_probability = temporal_evidence_probability(
            role_logits, base.temporal_role_mode
        )
        evidence_weights = evidence_probability[:, None]
        pooled = (fused * evidence_weights).sum(dim=-1) / evidence_weights.sum(
            dim=-1
        ).clamp_min(1e-5)
        no_evidence_logit = base.no_evidence_logit + self.no_evidence_delta_head(
            pooled
        ).squeeze(-1)
        return PromptComposition(
            semantic_condition=base.semantic_condition,
            role_logits=role_logits,
            evidence_probability=evidence_probability,
            no_evidence_logit=no_evidence_logit,
            frame_features=base.frame_features,
            frame_hop_samples=base.frame_hop_samples,
            temporal_role_mode=base.temporal_role_mode,
            semantic_separation_mode=base.semantic_separation_mode,
            anchor_semantic_condition=base.anchor_semantic_condition,
            answer_semantic_condition=base.answer_semantic_condition,
            same_semantic_logit=base.same_semantic_logit,
            foundation_semantic_candidate_weight=(
                base.foundation_semantic_candidate_weight
            ),
        )


class AudioSepConditionedAdapter(nn.Module):
    """Inject composer conditions into a separately installed AudioSep model.

    The adapter intentionally accepts ``ss_model`` rather than owning the
    third-party repository. This keeps licensing and checkpoint acquisition
    explicit and lets tests use a small compatible stand-in.
    """

    def __init__(
        self,
        ss_model: nn.Module,
        condition_dim: int = 512,
        freeze_separator: bool = True,
        separator_aware_refiner: SeparatorAwareTemporalRefiner | None = None,
    ) -> None:
        super().__init__()
        self.ss_model = ss_model
        self.condition_dim = condition_dim
        self.separator_aware_refiner = separator_aware_refiner
        self.deterministic_stft_patch = False
        if freeze_separator:
            self.ss_model.eval()
            for parameter in self.ss_model.parameters():
                parameter.requires_grad = False

    @classmethod
    def from_repository(
        cls,
        repository_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        device: torch.device | str,
        freeze_separator: bool = True,
        qces_config: QCESConfig | None = None,
    ) -> "AudioSepConditionedAdapter":
        """Load only the pretrained separator from the official AudioSep repo.

        QCES replaces AudioSep's CLAP query encoder with its learned prompt
        composer. Building only the official ResUNet avoids loading CLAP and
        Lightning, then extracts the ``ss_model`` weights from either the
        Hugging Face or PyTorch-Lightning checkpoint format.
        """

        repository_root = repository_root.resolve()
        config_path = config_path.resolve()
        checkpoint_path = checkpoint_path.resolve()
        for path in (repository_root, config_path, checkpoint_path):
            if not path.exists():
                raise FileNotFoundError(path)
        root_string = str(repository_root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        try:
            import yaml

            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            model_config = config["model"]
            from models.resunet import ResUNet30

            ss_model = ResUNet30(
                input_channels=int(model_config["input_channels"]),
                output_channels=int(model_config["output_channels"]),
                condition_size=int(model_config["condition_size"]),
            )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise TypeError("AudioSep checkpoint must contain a state dict")
            state = payload.get("state_dict", payload)
            if not isinstance(state, dict):
                raise TypeError("AudioSep state_dict is not a mapping")
            separator_state = {
                key.removeprefix("ss_model."): value
                for key, value in state.items()
                if key.startswith("ss_model.")
            }
            if not separator_state:
                separator_keys = set(ss_model.state_dict())
                separator_state = {
                    key: value for key, value in state.items() if key in separator_keys
                }
            if not separator_state:
                raise KeyError("checkpoint has no AudioSep ss_model weights")
            missing, unexpected = ss_model.load_state_dict(
                separator_state, strict=False
            )
            if missing or unexpected:
                raise RuntimeError(
                    "incompatible AudioSep separator checkpoint: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
            ss_model = ss_model.to(device).eval()
            deterministic_stft_patch = False
            if torch.are_deterministic_algorithms_enabled():
                stft = getattr(ss_model, "stft", None)
                if (
                    stft is not None
                    and bool(getattr(stft, "center", False))
                    and getattr(stft, "pad_mode", None) == "reflect"
                ):
                    stft.forward = MethodType(
                        _deterministic_torchlibrosa_stft_forward, stft
                    )
                    deterministic_stft_patch = True
        except Exception as exc:
            raise RuntimeError(
                "Could not load AudioSep's ResUNet separator. Install "
                "torchlibrosa and verify the official repository, config, "
                "and checkpoint paths."
            ) from exc
        adapter = cls(
            ss_model=ss_model,
            condition_dim=int(model_config["condition_size"]),
            freeze_separator=freeze_separator,
            separator_aware_refiner=(
                SeparatorAwareTemporalRefiner(qces_config)
                if qces_config is not None and qces_config.separator_aware_refiner
                else None
            ),
        )
        adapter.deterministic_stft_patch = deterministic_stft_patch
        return adapter

    def train(self, mode: bool = True) -> "AudioSepConditionedAdapter":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.ss_model.parameters()):
            self.ss_model.eval()
        return self

    def forward(
        self, mixture: torch.Tensor, composition: PromptComposition
    ) -> SeparationOutput:
        mixture = _mono_waveform(mixture)
        if composition.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE:
            return self._forward_dual_role(mixture, composition)
        if composition.semantic_separation_mode != UNION_SINGLE_SEMANTIC_MODE:
            raise ValueError(
                "unsupported semantic separation mode: "
                f"{composition.semantic_separation_mode!r}"
            )
        condition = composition.semantic_condition
        if condition.size(-1) != self.condition_dim:
            raise ValueError(
                f"AudioSep needs condition_dim={self.condition_dim}, "
                f"got {condition.size(-1)}"
            )
        result: Any = self.ss_model(
            {"mixture": mixture[:, None], "condition": condition}
        )
        raw_evidence = result["waveform"] if isinstance(result, dict) else result
        if raw_evidence.ndim == 3 and raw_evidence.size(1) == 1:
            raw_evidence = raw_evidence[:, 0]
        if raw_evidence.shape != mixture.shape:
            raise ValueError(
                f"AudioSep returned {tuple(raw_evidence.shape)}, expected "
                f"{tuple(mixture.shape)}"
            )
        refined_composition = composition
        if self.separator_aware_refiner is not None:
            refined_composition = self.separator_aware_refiner(
                mixture, raw_evidence, composition
            )
        temporal_gate = qces_linear_interpolate_1d(
            refined_composition.evidence_probability[:, None], mixture.size(-1)
        ).squeeze(1)
        sample_mask = temporal_gate
        evidence = raw_evidence * sample_mask
        residual = mixture - evidence
        mixture_error = (mixture - evidence - residual).abs().mean(dim=-1)
        return SeparationOutput(
            evidence=evidence,
            residual=residual,
            mask=sample_mask,
            raw_mask=None,
            raw_evidence=raw_evidence,
            mixture_error=mixture_error,
            refined_composition=(
                refined_composition
                if self.separator_aware_refiner is not None
                else None
            ),
            semantic_separation_mode=UNION_SINGLE_SEMANTIC_MODE,
            physical_separator_forwards_per_batch=1,
            effective_separator_evaluations_per_record=1,
        )

    def _forward_dual_role(
        self, mixture: torch.Tensor, composition: PromptComposition
    ) -> SeparationOutput:
        """Render learned anchor/answer conditions with shared-target routing.

        The frozen backbone receives a ``2B`` batch in one physical forward,
        which is nevertheless two effective separator evaluations per input
        record.  The differentiable same-semantic probability interpolates
        between distinct-source linear addition and one union-gated shared
        estimate.  It never makes a thresholded label/reuse decision.
        """

        if self.separator_aware_refiner is not None:
            raise ValueError(
                "dual_role semantic separation cannot use the legacy "
                "separator-aware refiner"
            )
        anchor_condition = composition.anchor_semantic_condition
        answer_condition = composition.answer_semantic_condition
        same_probability = composition.same_semantic_probability
        if (
            anchor_condition is None
            or answer_condition is None
            or same_probability is None
        ):
            raise ValueError(
                "dual_role separation requires anchor/answer conditions and "
                "a learned same-semantic logit"
            )
        for name, condition in (
            ("anchor", anchor_condition),
            ("answer", answer_condition),
        ):
            if condition.ndim != 2 or condition.shape != (
                mixture.size(0),
                self.condition_dim,
            ):
                raise ValueError(
                    f"{name} AudioSep condition must have shape "
                    f"[{mixture.size(0)}, {self.condition_dim}], got "
                    f"{tuple(condition.shape)}"
                )
        if same_probability.shape != (mixture.size(0),):
            raise ValueError("same-semantic probability must have one value per record")

        result: Any = self.ss_model(
            {
                "mixture": torch.cat([mixture, mixture], dim=0)[:, None],
                "condition": torch.cat([anchor_condition, answer_condition], dim=0),
            }
        )
        raw = result["waveform"] if isinstance(result, dict) else result
        if raw.ndim == 3 and raw.size(1) == 1:
            raw = raw[:, 0]
        expected_shape = (2 * mixture.size(0), mixture.size(1))
        if raw.shape != expected_shape:
            raise ValueError(
                f"AudioSep returned {tuple(raw.shape)}, expected {expected_shape}"
            )
        anchor_raw, answer_raw = raw.chunk(2, dim=0)

        probabilities = composition.role_probabilities
        anchor_gate = qces_linear_interpolate_1d(
            probabilities[..., ROLE_ANCHOR][:, None], mixture.size(-1)
        ).squeeze(1)
        answer_gate = qces_linear_interpolate_1d(
            probabilities[..., ROLE_ANSWER][:, None], mixture.size(-1)
        ).squeeze(1)
        union_gate = qces_linear_interpolate_1d(
            composition.evidence_probability[:, None], mixture.size(-1)
        ).squeeze(1)

        distinct_evidence = anchor_raw * anchor_gate + answer_raw * answer_gate
        # For equal semantic targets, both conditions should address the same
        # source class. Averaging their estimates and applying the role-union
        # gate counts that shared target once, including overlap samples.
        shared_evidence = 0.5 * (anchor_raw + answer_raw) * union_gate
        route = same_probability[:, None]
        evidence = (1.0 - route) * distinct_evidence + route * shared_evidence
        residual = mixture - evidence
        raw_distinct = anchor_raw + answer_raw
        raw_shared = 0.5 * (anchor_raw + answer_raw)
        raw_evidence = (1.0 - route) * raw_distinct + route * raw_shared
        mixture_error = (mixture - evidence - residual).abs().mean(dim=-1)
        return SeparationOutput(
            evidence=evidence,
            residual=residual,
            mask=union_gate,
            raw_mask=None,
            raw_evidence=raw_evidence,
            mixture_error=mixture_error,
            semantic_separation_mode=DUAL_ROLE_SEMANTIC_MODE,
            physical_separator_forwards_per_batch=1,
            effective_separator_evaluations_per_record=2,
            anchor_raw_evidence=anchor_raw,
            answer_raw_evidence=answer_raw,
            same_semantic_probability=same_probability,
        )
