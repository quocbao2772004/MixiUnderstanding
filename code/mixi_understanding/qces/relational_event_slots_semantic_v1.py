"""Semantic extension of frozen class-agnostic relational event slots."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
    _sinusoidal_position_encoding,
    center_width_to_intervals,
)


class RelationalEventSlotsSemanticV1(RelationalEventSlotsV1):
    """Attach an event-label classifier to each DETR interval query.

    The inherited temporal branch is byte-for-byte compatible with the v1
    checkpoint.  Query embeddings, rather than a pooled mixture interval, are
    classified so overlapping events can retain separate semantic identities.
    """

    def __init__(self, config: RelationalEventSlotsV1Config, num_classes: int) -> None:
        super().__init__(config)
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.num_classes = int(num_classes)
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, self.num_classes),
        )

    def forward(
        self, features: torch.Tensor, valid_frame_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.config.feature_dim:
            raise ValueError("features must have shape [B,T,feature_dim]")
        batch, frames, _ = features.shape
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch, frames, dtype=torch.bool, device=features.device)
        if valid_frame_mask.shape != (batch, frames):
            raise ValueError("valid_frame_mask must have shape [B,T]")
        valid_frame_mask = valid_frame_mask.bool()
        position = _sinusoidal_position_encoding(
            frames, self.config.hidden_dim, device=features.device, dtype=features.dtype
        )
        memory = self.input_projection(self.input_norm(features)) + position[None]
        memory = self.encoder(memory, src_key_padding_mask=~valid_frame_mask)
        queries = self.slot_queries.weight[None].expand(batch, -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=~valid_frame_mask)
        raw_box = self.box_head(decoded)
        reference_logits = torch.logit(self.reference_center.clamp(1e-4, 1.0 - 1e-4))[None]
        center = torch.sigmoid(reference_logits + raw_box[..., 0])
        width = torch.sigmoid(raw_box[..., 1])
        center_width = torch.stack((center, width), dim=-1)
        return {
            "objectness_logits": self.objectness_head(decoded).squeeze(-1),
            "center_width": center_width,
            "intervals": center_width_to_intervals(center_width),
            "frame_event_logits": self.frame_event_head(memory).squeeze(-1),
            "frame_onset_logits": self.frame_onset_head(memory).squeeze(-1),
            "slot_class_logits": self.semantic_head(decoded),
        }

