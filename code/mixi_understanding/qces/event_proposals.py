"""Separator-as-detector event proposals for QCES-v6.

A frozen text-queried separator is normally used once per question, to render
the answer.  Here it is queried once per *candidate label* first, and the set of
returned stems is read as a detection problem: a label is active in a frame
when its stem, and not a competing stem, carries the energy of that frame.

Two readers are provided.

``energy_activity``
    A training-free reader.  It is the honest lower bound of the idea and
    doubles as an ablation.

``ProposalHead``
    A small label-agnostic temporal network over the stem descriptors.  It sees
    only energy/onset/shape statistics and cross-label competition, never a
    label identity or a class index, so nothing about it is tied to the label
    set it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from mixi_understanding.qces.stem_features import (
    EPSILON,
    NUM_BANDS,
    FrameGrid,
)

HEAD_FEATURE_DIM = 11 + NUM_BANDS + 4


@dataclass(frozen=True)
class EventProposal:
    """One detected occurrence of a candidate label."""

    label: str
    onset_seconds: float
    offset_seconds: float
    confidence: float

    @property
    def duration_seconds(self) -> float:
        return self.offset_seconds - self.onset_seconds


def head_features(
    stem_features: torch.Tensor,
    mixture_features: torch.Tensor,
    clap_mixture_similarity: torch.Tensor | None = None,
    clap_stem_similarity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build ``[L, HEAD_FEATURE_DIM, T]`` inputs for a candidate label set.

    ``stem_features`` is ``[L, STEM_FEATURE_DIM, T]`` and ``mixture_features``
    is ``[STEM_FEATURE_DIM, T]``.  Every derived channel is a *relative*
    quantity so the representation is invariant to the overall level of the
    scene and equivariant to a permutation of the candidate labels.

    The two optional arguments carry the identity evidence.  A separator stem
    tells you *when* something was pulled out but not *whether* the thing pulled
    out is the queried class, and measured on this benchmark the stem energy
    alone separates present from absent labels at only about 0.60 AUC.  The
    shared audio-text embedding space of the separator's own query encoder
    supplies the missing identity channel: ``clap_mixture_similarity`` is a
    ``[L, T]`` windowed similarity between the mixture and each candidate label
    text, and ``clap_stem_similarity`` is an ``[L, L]`` matrix whose row ``i``
    holds the similarity between stem ``i`` and every candidate label text, so
    its diagonal asks whether a stem actually sounds like what was requested.
    """

    if stem_features.ndim != 3:
        raise ValueError("stem features must be [labels, features, frames]")
    stem_features = stem_features.float()
    mixture_features = mixture_features.float()
    num_labels, _, num_frames = stem_features.shape

    energy = stem_features[:, 0]
    flux = stem_features[:, 1]
    bands = stem_features[:, 2 : 2 + NUM_BANDS]
    mixture_energy = mixture_features[0]
    mixture_flux = mixture_features[1]

    log_energy = torch.log10(energy + EPSILON)
    log_mixture = torch.log10(mixture_energy + EPSILON)
    # Level-invariant channels.
    relative = log_energy - log_mixture[None]
    total = energy.sum(dim=0, keepdim=True)
    share = energy / total.clamp_min(EPSILON)
    peak = energy.amax(dim=1, keepdim=True).clamp_min(EPSILON)
    envelope = energy / peak
    if num_labels > 1:
        top2 = energy.topk(2, dim=0).values
        strongest_other = torch.where(
            energy >= top2[0:1], top2[1:2], top2[0:1]
        ).expand_as(energy)
        margin = log_energy - torch.log10(strongest_other + EPSILON)
        rank = (energy[:, None] < energy[None, :]).float().sum(dim=1) / max(
            num_labels - 1, 1
        )
    else:
        margin = torch.zeros_like(log_energy)
        rank = torch.zeros_like(log_energy)

    position = (
        torch.arange(num_frames, device=stem_features.device, dtype=torch.float32)
        / max(num_frames - 1, 1)
    ).expand(num_labels, num_frames)
    scaled_log_energy = (log_energy + 8.0) / 4.0
    scaled_log_mixture = ((log_mixture + 8.0) / 4.0)[None].expand_as(scaled_log_energy)

    if clap_mixture_similarity is None:
        similarity = torch.zeros_like(log_energy)
        similarity_share = torch.zeros_like(log_energy)
    else:
        similarity = clap_mixture_similarity.float().to(stem_features.device)
        # Scaled so the useful part of a cosine similarity range occupies most
        # of the unit interval before the head's input normalisation sees it.
        similarity_share = torch.softmax(similarity * 20.0, dim=0)
        similarity = similarity * 10.0

    if clap_stem_similarity is None:
        self_similarity = torch.zeros_like(log_energy)
        self_margin = torch.zeros_like(log_energy)
    else:
        matrix = clap_stem_similarity.float().to(stem_features.device)
        diagonal = torch.diagonal(matrix)
        if num_labels > 1:
            off_diagonal = matrix - torch.diag(
                torch.full_like(diagonal, float("inf"))
            )
            best_other = off_diagonal.amax(dim=1)
        else:
            best_other = torch.zeros_like(diagonal)
        self_similarity = (diagonal * 10.0)[:, None].expand_as(log_energy)
        self_margin = ((diagonal - best_other) * 10.0)[:, None].expand_as(log_energy)

    channels = [
        scaled_log_energy,
        relative,
        share,
        envelope,
        margin,
        rank,
        flux,
        mixture_flux[None].expand_as(flux),
        scaled_log_mixture,
        position,
        share * envelope,
        similarity,
        similarity_share,
        self_similarity,
        self_margin,
    ]
    stacked = torch.stack(channels, dim=1)
    return torch.cat([stacked, bands], dim=1)


# Channel groups of :func:`head_features`, in the order the channels are built.
# Zeroing a group is how the paper's feature ablation is run: the head keeps its
# shape, so a single training recipe covers every ablation arm.
FEATURE_GROUPS: Mapping[str, tuple[int, ...]] = {
    "clap": (11, 12, 13, 14),
    "clap_stem": (13, 14),
    "clap_mixture": (11, 12),
    "competition": (2, 4, 5, 10, 12),
    "spectral_shape": tuple(range(15, 15 + NUM_BANDS)),
    "onset_strength": (6,),
}


def zero_feature_groups(
    features: torch.Tensor, groups: Sequence[str]
) -> torch.Tensor:
    """Return ``features`` with every channel of the named groups set to zero."""

    if not groups:
        return features
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"unknown feature groups: {unknown}")
    masked = features.clone()
    for group in groups:
        for channel in FEATURE_GROUPS[group]:
            masked[:, channel] = 0.0
    return masked


def features_from_cache(
    entry: Mapping[str, Any], labels: Sequence[str] | None = None
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    """Read one cached scene into ``(labels, head features, stem features)``.

    ``labels`` selects and orders a candidate subset; the cross-label channels
    are then computed over exactly that subset, which is what makes the cache
    reusable across every question that shares the scene.
    """

    available = list(entry["labels"])
    selected = list(available) if labels is None else [
        label for label in labels if label in available
    ]
    index = {label: position for position, label in enumerate(available)}
    positions = [index[label] for label in selected]
    stems = entry["stems"].float()[positions]
    mixture = entry["mixture"].float()
    clap_mixture = entry.get("clap_mixture_similarity")
    clap_stem = entry.get("clap_stem_similarity")
    if clap_mixture is not None:
        clap_mixture = clap_mixture.float()[positions]
    if clap_stem is not None:
        clap_stem = clap_stem.float()[positions][:, positions]
    return (
        tuple(selected),
        head_features(stems, mixture, clap_mixture, clap_stem),
        stems,
    )


def energy_activity(
    stem_features: torch.Tensor,
    mixture_features: torch.Tensor,
    *,
    share_weight: float = 1.0,
) -> torch.Tensor:
    """Training-free activity in ``[0, 1]``: competition-normalised envelope."""

    features = head_features(stem_features, mixture_features)
    share = features[:, 2]
    envelope = features[:, 3]
    return (share.clamp(0.0, 1.0) ** share_weight) * envelope.clamp(0.0, 1.0)


class ProposalHead(nn.Module):
    """Dilated temporal convolutions shared by every candidate label.

    Two outputs are produced per candidate label: a per-frame activity logit
    and one utterance-level presence logit.  The presence branch exists because
    the benchmark's ``no_evidence`` questions are exactly the questions whose
    named label is absent from the scene, and because a dilated stack with a
    2.5-second receptive field cannot otherwise tell a quiet true event from
    steady separator leakage on an absent label.  A global-context vector
    computed from the whole clip is broadcast back into the frame pathway, so
    the frame decision is also allowed to depend on clip-level evidence.
    """

    def __init__(
        self,
        input_dim: int = HEAD_FEATURE_DIM,
        channels: int = 96,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.stem = nn.Conv1d(input_dim, channels, kernel_size=1)
        blocks = []
        for dilation in dilations:
            blocks.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            channels,
                            channels,
                            kernel_size=5,
                            padding=2 * dilation,
                            dilation=dilation,
                        ),
                        "norm": nn.GroupNorm(8, channels),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.context = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.presence = nn.Linear(channels, 1)
        self.output = nn.Conv1d(channels, 1, kernel_size=1)
        # A separate onset head, because a question like "the third Camera"
        # needs occurrence *count*, and two adjacent same-label events merge
        # into one activity region that no threshold can split.
        self.onset = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[L, D, T]`` features to frame logits, onset logits and presence."""

        hidden = self.stem(self.input_norm(features))
        for block in self.blocks:
            residual = hidden
            hidden = block["conv"](hidden)
            hidden = block["norm"](hidden)
            hidden = torch.nn.functional.gelu(hidden)
            hidden = self.dropout(hidden)
            hidden = hidden + residual
        pooled = torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=-1)
        context = self.context(pooled)
        presence = self.presence(context).squeeze(-1)
        hidden = hidden + context[..., None]
        return (
            self.output(hidden).squeeze(1),
            self.onset(hidden).squeeze(1),
            presence,
        )


def median_filter(activity: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return activity
    if kernel % 2 == 0:
        kernel += 1
    padded = torch.nn.functional.pad(
        activity[:, None], (kernel // 2, kernel // 2), mode="replicate"
    )
    return padded.unfold(-1, kernel, 1).median(dim=-1).values.squeeze(1)


def _local_maxima(curve: torch.Tensor, threshold: float, distance: int) -> torch.Tensor:
    """Boolean mask of peaks that dominate a ``+/- distance`` neighbourhood.

    Vectorised with a max pool so that a full threshold sweep over a split stays
    cheap; the equivalent Python loop dominated training time.
    """

    window = 2 * distance + 1
    pooled = torch.nn.functional.max_pool1d(
        curve[None, None], kernel_size=window, stride=1, padding=distance
    )[0, 0]
    return (curve >= pooled) & (curve >= threshold)


def _runs(active: torch.Tensor) -> list[tuple[int, int]]:
    """Half-open index ranges of the True runs in a 1-D boolean tensor."""

    if not bool(active.any()):
        return []
    padded = torch.cat(
        [
            torch.zeros(1, dtype=torch.bool),
            active,
            torch.zeros(1, dtype=torch.bool),
        ]
    )
    edges = padded[1:].int() - padded[:-1].int()
    starts = torch.nonzero(edges == 1, as_tuple=False).flatten().tolist()
    stops = torch.nonzero(edges == -1, as_tuple=False).flatten().tolist()
    return list(zip(starts, stops))


def decode_proposals(
    labels: Sequence[str],
    activity: torch.Tensor,
    grid: FrameGrid,
    *,
    threshold: float = 0.5,
    median_kernel: int = 5,
    minimum_duration_seconds: float = 0.10,
    merge_gap_seconds: float = 0.06,
    minimum_confidence: float = 0.0,
    onset_activity: torch.Tensor | None = None,
    onset_threshold: float = 0.5,
    onset_distance_seconds: float = 0.25,
) -> list[EventProposal]:
    """Turn per-label frame activity into onset/offset/confidence proposals.

    When ``onset_activity`` is supplied, an active region holding more than one
    onset peak is split at the midpoints between consecutive peaks.  Without
    that split, two adjacent occurrences of the same label become one proposal
    and every ordinal after them shifts by one.
    """

    if activity.ndim != 2 or activity.shape[0] != len(labels):
        raise ValueError("activity must be [labels, frames] aligned to labels")
    smoothed = median_filter(activity.detach().float().cpu(), median_kernel)
    # The onset curve is deliberately not median filtered: an onset is a peak,
    # and a median filter removes narrow peaks, which is exactly the evidence
    # that two occurrences are present.  Peak picking below already enforces a
    # minimum spacing.
    onsets = None if onset_activity is None else onset_activity.detach().float().cpu()
    merge_frames = max(0, grid.seconds_to_frame(merge_gap_seconds))
    minimum_frames = max(1, grid.seconds_to_frame(minimum_duration_seconds))
    onset_distance = max(1, grid.seconds_to_frame(onset_distance_seconds))

    peak_mask = (
        None
        if onsets is None
        else torch.stack(
            [
                _local_maxima(onsets[index], onset_threshold, onset_distance)
                for index in range(onsets.shape[0])
            ]
        )
    )

    proposals: list[EventProposal] = []
    for index, label in enumerate(labels):
        runs = _runs(smoothed[index] >= threshold)
        spans: list[list[int]] = []
        for start, stop in runs:
            if spans and start - spans[-1][1] <= merge_frames:
                spans[-1][1] = stop
            else:
                spans.append([start, stop])

        refined: list[tuple[int, int]] = []
        for start, stop in spans:
            peaks = (
                []
                if peak_mask is None
                else torch.nonzero(peak_mask[index, start:stop], as_tuple=False)
                .flatten()
                .add(start)
                .tolist()
            )
            # Peaks closer than the minimum spacing are one onset seen twice.
            spaced: list[int] = []
            for peak in peaks:
                if spaced and peak - spaced[-1] < onset_distance:
                    if float(onsets[index, peak]) > float(onsets[index, spaced[-1]]):
                        spaced[-1] = peak
                    continue
                spaced.append(peak)
            if len(spaced) <= 1:
                refined.append((start, stop))
                continue
            boundaries = [start]
            for left, right in zip(spaced, spaced[1:]):
                boundaries.append((left + right) // 2)
            boundaries.append(stop)
            refined.extend(zip(boundaries, boundaries[1:]))

        for start, stop in refined:
            if stop - start < minimum_frames:
                continue
            confidence = float(smoothed[index, start:stop].mean())
            if confidence < minimum_confidence:
                continue
            proposals.append(
                EventProposal(
                    label=label,
                    onset_seconds=grid.frame_to_seconds(start),
                    offset_seconds=grid.frame_to_seconds(stop),
                    confidence=confidence,
                )
            )
    proposals.sort(key=lambda item: (item.onset_seconds, item.label))
    return proposals
