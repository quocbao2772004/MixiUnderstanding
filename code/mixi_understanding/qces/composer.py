"""Role-aware semantic and temporal prompt composition from audio and question."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    CONVEX_SEMANTIC_INTERPOLATION,
    DUAL_ROLE_SEMANTIC_MODE,
    LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
    LEGACY_TEMPORAL_ROLE_MODE,
    NO_FOUNDATION_FEATURES,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    QUESTION_RESIDUAL_SEMANTIC_MIXING,
    QCESConfig,
    UNION_SINGLE_SEMANTIC_MODE,
    TEMPORAL_ROLE_MODES,
)
from mixi_understanding.qces.signal import (
    qces_linear_interpolate_1d,
    qces_stft,
)


ROLE_NONE = 0
ROLE_ANCHOR = 1
ROLE_ANSWER = 2
ROLE_NAMES = ("none", "anchor", "answer")
FOUNDATION_CLAP_DIM = 512
FOUNDATION_CLAP_FRAMES = 32


def temporal_role_probabilities(
    role_logits: torch.Tensor, temporal_role_mode: str
) -> torch.Tensor:
    """Map the shape-compatible three-logit head to role probabilities.

    ``exclusive_softmax`` is the historical behavior. In
    ``independent_sigmoid`` all three frame labels are binary, so anchor and
    answer may both be active. Keeping the head shape unchanged lets legacy
    checkpoints retain their exact parameter tensors while the checkpointed
    config makes the interpretation explicit.
    """

    if role_logits.size(-1) != len(ROLE_NAMES):
        raise ValueError(
            f"role logits need {len(ROLE_NAMES)} channels, got "
            f"{role_logits.size(-1)}"
        )
    if temporal_role_mode == LEGACY_TEMPORAL_ROLE_MODE:
        return role_logits.softmax(dim=-1)
    if temporal_role_mode == OVERLAP_AWARE_TEMPORAL_ROLE_MODE:
        return role_logits.sigmoid()
    raise ValueError(
        f"temporal_role_mode must be one of {TEMPORAL_ROLE_MODES}, got "
        f"{temporal_role_mode!r}"
    )


def temporal_evidence_probability(
    role_logits: torch.Tensor, temporal_role_mode: str
) -> torch.Tensor:
    """Return the evidence-union probability for either role formulation."""

    probabilities = temporal_role_probabilities(role_logits, temporal_role_mode)
    if temporal_role_mode == LEGACY_TEMPORAL_ROLE_MODE:
        return 1.0 - probabilities[..., ROLE_NONE]
    anchor = probabilities[..., ROLE_ANCHOR]
    answer = probabilities[..., ROLE_ANSWER]
    # Probabilistic OR retains both gradients in an overlap frame and is
    # bounded without an artificial anchor-vs-answer competition.
    return 1.0 - (1.0 - anchor) * (1.0 - answer)


def validate_foundation_clap_features(
    question_clap: torch.Tensor,
    scene_clap: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    """Validate inference-safe frozen CLAP inputs without semantic metadata."""

    if question_clap.shape != (batch_size, FOUNDATION_CLAP_DIM):
        raise ValueError(
            "question_clap must have shape "
            f"[{batch_size}, {FOUNDATION_CLAP_DIM}], got "
            f"{tuple(question_clap.shape)}"
        )
    if scene_clap.shape != (
        batch_size,
        FOUNDATION_CLAP_FRAMES,
        FOUNDATION_CLAP_DIM,
    ):
        raise ValueError(
            "scene_clap must have shape "
            f"[{batch_size}, {FOUNDATION_CLAP_FRAMES}, {FOUNDATION_CLAP_DIM}], "
            f"got {tuple(scene_clap.shape)}"
        )
    for name, tensor in (("question_clap", question_clap), ("scene_clap", scene_clap)):
        if not tensor.dtype.is_floating_point:
            raise ValueError(f"{name} must be floating point")
        if tensor.device != device:
            raise ValueError(
                f"{name} device {tensor.device} differs from waveform device {device}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains NaN or Inf")
    question_norms = torch.linalg.vector_norm(question_clap.float(), dim=-1)
    scene_norms = torch.linalg.vector_norm(scene_clap.float(), dim=-1)
    if not torch.allclose(
        question_norms,
        torch.ones_like(question_norms),
        rtol=1e-3,
        atol=1e-3,
    ):
        raise ValueError("question_clap must be L2 normalized")
    if not torch.allclose(
        scene_norms,
        torch.ones_like(scene_norms),
        rtol=2e-3,
        atol=2e-3,
    ):
        raise ValueError("every scene_clap frame must be L2 normalized")


def role_pool_scene_clap(
    scene_clap: torch.Tensor,
    role_weights: torch.Tensor,
) -> torch.Tensor:
    """Pool acoustic CLAP frames with learned temporal-role probabilities."""

    if scene_clap.ndim != 3 or scene_clap.shape[1:] != (
        FOUNDATION_CLAP_FRAMES,
        FOUNDATION_CLAP_DIM,
    ):
        raise ValueError("scene_clap must have shape [B, 32, 512]")
    if role_weights.ndim != 2 or role_weights.size(0) != scene_clap.size(0):
        raise ValueError("role weights must have shape [B, T]")
    aligned_weights = qces_linear_interpolate_1d(
        role_weights[:, None].float(),
        FOUNDATION_CLAP_FRAMES,
    ).squeeze(1)
    aligned_weights = aligned_weights.to(scene_clap.dtype).clamp_min(0.0)
    pooled = (scene_clap * aligned_weights.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / aligned_weights.sum(dim=1, keepdim=True).clamp_min(1e-5)
    return F.normalize(pooled.float(), dim=-1)


def bounded_semantic_residual(
    base: torch.Tensor,
    delta: torch.Tensor,
    scale_logit: torch.Tensor,
) -> torch.Tensor:
    """Add a learned unit residual with a sigmoid-bounded global scale."""

    if base.shape != delta.shape or base.ndim != 2:
        raise ValueError("semantic base and delta must share shape [B, D]")
    scale = scale_logit.sigmoid()
    return F.normalize(base + scale * F.normalize(delta, dim=-1), dim=-1)


def foundation_semantic_mix(
    base: torch.Tensor,
    candidate: torch.Tensor,
    gate_logit: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Mix an inference-safe base and learned condition under a checkpoint mode.

    The historical bounded residual has a candidate/base magnitude ratio of at
    most one.  Convex interpolation instead has the acoustic endpoint at gate
    zero and the fully learned endpoint at gate one, making every unit learned
    condition reachable without an unbounded parameterization.
    """

    if mode == LEGACY_BOUNDED_SEMANTIC_RESIDUAL:
        return bounded_semantic_residual(base, candidate, gate_logit)
    if mode == CONVEX_SEMANTIC_INTERPOLATION:
        if base.shape != candidate.shape or base.ndim != 2:
            raise ValueError("semantic base and candidate must share shape [B, D]")
        gate = gate_logit.sigmoid()
        learned = F.normalize(candidate, dim=-1)
        return F.normalize((1.0 - gate) * base + gate * learned, dim=-1)
    if mode == QUESTION_RESIDUAL_SEMANTIC_MIXING:
        if base.shape != candidate.shape or base.ndim != 2:
            raise ValueError("semantic base and candidate must share shape [B, D]")
        # ``base`` is the official normalized full-question CLAP embedding in
        # this mode.  The learned delta is intentionally not norm-bounded: it
        # must be able to cancel/replace a question prior when the answer source
        # is implicit and can only be inferred from the mixture.
        return F.normalize(base + candidate, dim=-1)
    raise ValueError(f"unsupported foundation semantic mixing mode: {mode!r}")


@dataclass
class PromptComposition:
    """Continuous conditions produced for one audio/question batch."""

    semantic_condition: torch.Tensor
    role_logits: torch.Tensor
    evidence_probability: torch.Tensor
    no_evidence_logit: torch.Tensor
    frame_features: torch.Tensor
    frame_hop_samples: int
    temporal_role_mode: str = LEGACY_TEMPORAL_ROLE_MODE
    semantic_separation_mode: str = UNION_SINGLE_SEMANTIC_MODE
    anchor_semantic_condition: torch.Tensor | None = None
    answer_semantic_condition: torch.Tensor | None = None
    same_semantic_logit: torch.Tensor | None = None
    # One checkpointed global mixer parameter, exported as a probability for
    # optimization diagnosis.  This is descriptive rather than a performance
    # metric: a larger value means more reliance on the learned candidate.
    foundation_semantic_candidate_weight: torch.Tensor | None = None

    @property
    def role_probabilities(self) -> torch.Tensor:
        return temporal_role_probabilities(self.role_logits, self.temporal_role_mode)

    @property
    def same_semantic_probability(self) -> torch.Tensor | None:
        if self.same_semantic_logit is None:
            return None
        return self.same_semantic_logit.sigmoid()


class QuestionEncoder(nn.Module):
    def __init__(self, config: QCESConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.question_dim, padding_idx=0
        )
        self.position = nn.Embedding(config.max_question_tokens, config.question_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.question_dim,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.question_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.question_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(config.question_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("question IDs and mask must both have shape [B, L]")
        if input_ids.size(1) > self.position.num_embeddings:
            raise ValueError("question exceeds configured max_question_tokens")
        if not attention_mask.any(dim=1).all():
            raise ValueError("every question needs at least one non-padding token")

        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        features = self.embedding(input_ids) + self.position(positions)[None]
        features = self.encoder(features, src_key_padding_mask=~attention_mask)
        features = self.output_norm(features)
        weights = attention_mask.to(features.dtype).unsqueeze(-1)
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return features, pooled


class AcousticFrameEncoder(nn.Module):
    """Encode a waveform into frame-synchronous features at the STFT hop rate."""

    def __init__(self, config: QCESConfig) -> None:
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.register_buffer(
            "window", torch.hann_window(config.win_length), persistent=False
        )
        frequency_bins = config.n_fft // 2 + 1
        self.network = nn.Sequential(
            nn.Conv1d(frequency_bins, config.audio_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8 if config.audio_dim % 8 == 0 else 1, config.audio_dim),
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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [B, N] or [B, 1, N]")
        spectrum = qces_stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=True,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        log_magnitude = torch.log1p(magnitude)
        scale = log_magnitude.square().mean(dim=(1, 2), keepdim=True).sqrt()
        normalized = log_magnitude / scale.clamp_min(1e-5)
        return self.network(normalized).transpose(1, 2)


class RoleAwarePromptComposer(nn.Module):
    """Compose a global separator condition and frame-level evidence roles."""

    def __init__(self, config: QCESConfig) -> None:
        super().__init__()
        self.config = config
        self.question_encoder = QuestionEncoder(config)
        self.audio_encoder = AcousticFrameEncoder(config)
        self.question_to_audio = nn.Linear(config.question_dim, config.audio_dim)
        self.question_tokens_to_audio = nn.Linear(config.question_dim, config.audio_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.audio_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(config.audio_dim)
        self.temporal_refiner = nn.Sequential(
            nn.Conv1d(config.audio_dim, config.audio_dim, 5, padding=2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.audio_dim, config.audio_dim, 5, padding=2),
        )
        self.role_head = nn.Linear(config.audio_dim, len(ROLE_NAMES))
        joint_dim = config.audio_dim + config.question_dim
        self.semantic_head = nn.Sequential(
            nn.Linear(joint_dim, 2 * config.condition_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.condition_dim, config.condition_dim),
        )
        if (
            config.foundation_semantic_mixing_mode
            == QUESTION_RESIDUAL_SEMANTIC_MIXING
        ):
            # Exact identity initialization: before learning, the AudioSep
            # condition is the official full-question CLAP embedding.  Only the
            # final layer is zeroed, so it immediately receives gradients and
            # then unlocks the preceding audio-question adapter.
            nn.init.zeros_(self.semantic_head[-1].weight)
            nn.init.zeros_(self.semantic_head[-1].bias)
        self.no_evidence_head = nn.Sequential(
            nn.Linear(joint_dim, config.audio_dim),
            nn.GELU(),
            nn.Linear(config.audio_dim, 1),
        )
        if config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES:
            self.foundation_question_projection = nn.Linear(
                FOUNDATION_CLAP_DIM, config.question_dim
            )
            self.foundation_question_norm = nn.LayerNorm(config.question_dim)
            self.foundation_scene_projection = nn.Linear(
                FOUNDATION_CLAP_DIM, config.audio_dim
            )
            # Preserve the historical 0.119 initialization for legacy
            # checkpoints.  The reachable convex mixer is opt-in and starts
            # halfway between its endpoints: fine-grained acoustic CLAP can be
            # nearly orthogonal to an AudioSep text-condition target, so a
            # 0.119 gate needlessly attenuates the only gradient that can learn
            # the question-conditioned endpoint.  No legacy checkpoint used
            # the convex mode.
            initial_semantic_gate_logit = (
                0.0
                if config.foundation_semantic_mixing_mode
                == CONVEX_SEMANTIC_INTERPOLATION
                else -2.0
            )
            self.foundation_semantic_residual_scale_logit = nn.Parameter(
                torch.tensor(initial_semantic_gate_logit)
            )
        # Keep every module shared by union_single and dual_role above this
        # candidate-only head.  Re-seeding before constructing either mode now
        # gives bit-identical common parameters, which is required for a fair
        # architecture ablation.  A legacy union composer still has exactly its
        # historical state-dict keys because assigning None registers nothing.
        self.same_semantic_head: nn.Module | None = None
        if config.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE:
            self.same_semantic_head = nn.Sequential(
                nn.Linear(2 * config.audio_dim + config.question_dim, config.audio_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.audio_dim, 1),
            )

    def forward(
        self,
        waveform: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        *,
        question_clap: torch.Tensor | None = None,
        scene_clap: torch.Tensor | None = None,
    ) -> PromptComposition:
        question_tokens, question_pooled = self.question_encoder(
            question_ids, question_mask
        )
        audio_frames = self.audio_encoder(waveform)
        foundation_scene = None
        if self.config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES:
            if question_clap is None or scene_clap is None:
                raise ValueError(
                    "audiosep_clap mode requires offline question_clap and "
                    "scene_clap features"
                )
            validate_foundation_clap_features(
                question_clap,
                scene_clap,
                batch_size=waveform.size(0),
                device=waveform.device,
            )
            question_feature = self.foundation_question_projection(
                question_clap.to(question_pooled.dtype)
            )
            question_tokens = self.foundation_question_norm(
                question_tokens + question_feature[:, None]
            )
            question_pooled = self.foundation_question_norm(
                question_pooled + question_feature
            )
            foundation_scene = scene_clap.to(audio_frames.dtype)
            projected_scene = self.foundation_scene_projection(foundation_scene)
            projected_scene = qces_linear_interpolate_1d(
                projected_scene.transpose(1, 2),
                audio_frames.size(1),
            ).transpose(1, 2)
            audio_frames = audio_frames + projected_scene
        elif self.config.foundation_feature_mode == NO_FOUNDATION_FEATURES:
            if question_clap is not None or scene_clap is not None:
                raise ValueError(
                    "foundation CLAP features were supplied while "
                    "foundation_feature_mode='none'"
                )
        else:
            raise ValueError(
                "unsupported foundation feature mode: "
                f"{self.config.foundation_feature_mode!r}"
            )
        audio_frames = audio_frames + self.question_to_audio(question_pooled)[:, None]
        question_memory = self.question_tokens_to_audio(question_tokens)
        attended, _ = self.cross_attention(
            query=audio_frames,
            key=question_memory,
            value=question_memory,
            key_padding_mask=~question_mask,
            need_weights=False,
        )
        fused = self.fusion_norm(audio_frames + attended)
        refined = self.temporal_refiner(fused.transpose(1, 2)).transpose(1, 2)
        fused = self.fusion_norm(fused + refined)

        role_logits = self.role_head(fused)
        evidence_probability = temporal_evidence_probability(
            role_logits, self.config.temporal_role_mode
        )

        evidence_weights = evidence_probability.unsqueeze(-1)
        evidence_pooled = (fused * evidence_weights).sum(dim=1) / evidence_weights.sum(
            dim=1
        ).clamp_min(1e-5)
        joint = torch.cat([evidence_pooled, question_pooled], dim=-1)
        anchor_semantic_condition = None
        answer_semantic_condition = None
        same_semantic_logit = None
        foundation_semantic_candidate_weight = (
            self.foundation_semantic_residual_scale_logit.sigmoid()
            if self.config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES
            and self.config.foundation_semantic_mixing_mode
            != QUESTION_RESIDUAL_SEMANTIC_MIXING
            else None
        )
        if self.config.semantic_separation_mode == UNION_SINGLE_SEMANTIC_MODE:
            if self.config.foundation_feature_mode == NO_FOUNDATION_FEATURES:
                # Keep this operation sequence unchanged for exact legacy
                # checkpoint/output regression.
                semantic_condition = F.normalize(self.semantic_head(joint), dim=-1)
            else:
                if foundation_scene is None:
                    raise RuntimeError("foundation scene features were not prepared")
                semantic_base = (
                    question_clap.to(joint.dtype)
                    if self.config.foundation_semantic_mixing_mode
                    == QUESTION_RESIDUAL_SEMANTIC_MIXING
                    else role_pool_scene_clap(
                        foundation_scene, evidence_probability
                    )
                )
                semantic_condition = foundation_semantic_mix(
                    semantic_base,
                    self.semantic_head(joint),
                    self.foundation_semantic_residual_scale_logit,
                    self.config.foundation_semantic_mixing_mode,
                )
        elif self.config.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE:
            probabilities = temporal_role_probabilities(
                role_logits, self.config.temporal_role_mode
            )
            anchor_weights = probabilities[..., ROLE_ANCHOR].unsqueeze(-1)
            answer_weights = probabilities[..., ROLE_ANSWER].unsqueeze(-1)
            anchor_pooled = (fused * anchor_weights).sum(dim=1) / (
                anchor_weights.sum(dim=1).clamp_min(1e-5)
            )
            answer_pooled = (fused * answer_weights).sum(dim=1) / (
                answer_weights.sum(dim=1).clamp_min(1e-5)
            )
            anchor_joint = torch.cat([anchor_pooled, question_pooled], dim=-1)
            answer_joint = torch.cat([answer_pooled, question_pooled], dim=-1)
            # A shared mapping is deliberate: role identity comes from the
            # independently pooled frames, while equal acoustic semantics are
            # encouraged to map to the same frozen-AudioSep condition space.
            if self.config.foundation_feature_mode == NO_FOUNDATION_FEATURES:
                anchor_semantic_condition = F.normalize(
                    self.semantic_head(anchor_joint), dim=-1
                )
                answer_semantic_condition = F.normalize(
                    self.semantic_head(answer_joint), dim=-1
                )
            else:
                if foundation_scene is None:
                    raise RuntimeError("foundation scene features were not prepared")
                # The new question-residual mode uses only the same full
                # question available at inference; all historical modes retain
                # role-pooled acoustic bases. No branch consumes answer labels,
                # oracle prompts, or event annotations.
                if (
                    self.config.foundation_semantic_mixing_mode
                    == QUESTION_RESIDUAL_SEMANTIC_MIXING
                ):
                    anchor_base = question_clap.to(anchor_joint.dtype)
                    answer_base = question_clap.to(answer_joint.dtype)
                else:
                    anchor_base = role_pool_scene_clap(
                        foundation_scene, probabilities[..., ROLE_ANCHOR]
                    )
                    answer_base = role_pool_scene_clap(
                        foundation_scene, probabilities[..., ROLE_ANSWER]
                    )
                anchor_semantic_condition = foundation_semantic_mix(
                    anchor_base,
                    self.semantic_head(anchor_joint),
                    self.foundation_semantic_residual_scale_logit,
                    self.config.foundation_semantic_mixing_mode,
                )
                answer_semantic_condition = foundation_semantic_mix(
                    answer_base,
                    self.semantic_head(answer_joint),
                    self.foundation_semantic_residual_scale_logit,
                    self.config.foundation_semantic_mixing_mode,
                )
            semantic_condition = F.normalize(
                anchor_semantic_condition + answer_semantic_condition, dim=-1
            )
            if self.same_semantic_head is None:
                raise RuntimeError("dual_role composer lacks same-semantic head")
            symmetric_pair = torch.cat(
                [
                    (anchor_pooled - answer_pooled).abs(),
                    anchor_pooled * answer_pooled,
                    question_pooled,
                ],
                dim=-1,
            )
            same_semantic_logit = self.same_semantic_head(symmetric_pair).squeeze(-1)
        else:  # QCESConfig validates this; retain a fail-closed local guard.
            raise ValueError(
                "unsupported semantic separation mode: "
                f"{self.config.semantic_separation_mode!r}"
            )
        no_evidence_logit = self.no_evidence_head(joint).squeeze(-1)
        return PromptComposition(
            semantic_condition=semantic_condition,
            role_logits=role_logits,
            evidence_probability=evidence_probability,
            no_evidence_logit=no_evidence_logit,
            frame_features=fused,
            frame_hop_samples=self.config.hop_length,
            temporal_role_mode=self.config.temporal_role_mode,
            semantic_separation_mode=self.config.semantic_separation_mode,
            anchor_semantic_condition=anchor_semantic_condition,
            answer_semantic_condition=answer_semantic_condition,
            same_semantic_logit=same_semantic_logit,
            foundation_semantic_candidate_weight=(foundation_semantic_candidate_weight),
        )
