"""Frame-level features of a frozen-separator stem.

The QCES-v6 proposal module never looks at a raw waveform.  It reads a compact
per-frame description of what the frozen text-queried separator returned for a
candidate label, so a whole benchmark split can be cached once and every
proposal ablation re-run without touching the GPU again.

The stored description is deliberately label-agnostic: it contains energy,
onset strength and a coarse spectral shape of the stem, never the identity of
the label that produced it.  A head trained on these features therefore has no
way to memorise a closed label set, which is what lets it transfer to the
held-out labels of the label-OOD split.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FRAME_HOP = 640
FRAME_FFT = 1024
NUM_FRAMES = 500
NUM_BANDS = 8
# energy, onset strength, then NUM_BANDS spectral-shape fractions
STEM_FEATURE_DIM = 2 + NUM_BANDS
EPSILON = 1e-10


@dataclass(frozen=True)
class FrameGrid:
    """The fixed analysis grid shared by every QCES-v6 component."""

    sample_rate: int
    num_frames: int = NUM_FRAMES
    hop: int = FRAME_HOP

    @property
    def frame_seconds(self) -> float:
        return self.hop / self.sample_rate

    def seconds_to_frame(self, seconds: float) -> int:
        return int(round(seconds / self.frame_seconds))

    def frame_to_seconds(self, frame: float) -> float:
        return float(frame) * self.frame_seconds


def _band_edges(num_bins: int, num_bands: int = NUM_BANDS) -> list[tuple[int, int]]:
    """Log-spaced bin groups, coarse at low frequency and wide at the top."""

    edges: list[tuple[int, int]] = []
    ratio = (num_bins / 4.0) ** (1.0 / num_bands)
    start = 1
    for index in range(num_bands):
        stop = int(round(4.0 * ratio ** (index + 1)))
        stop = min(num_bins, max(stop, start + 1))
        if index == num_bands - 1:
            stop = num_bins
        edges.append((start, stop))
        start = stop
    return edges


def waveform_frame_features(
    waveform: torch.Tensor, *, num_frames: int = NUM_FRAMES
) -> torch.Tensor:
    """Return ``[STEM_FEATURE_DIM, num_frames]`` features for one waveform.

    Row 0 is total frame energy, row 1 is positive spectral flux, and the
    remaining rows are the fraction of frame energy inside each log-spaced
    band.  Energies stay linear here so that a downstream consumer can pick its
    own normalisation without an irreversible log baked into the cache.
    """

    if waveform.ndim != 1:
        raise ValueError(f"expected a mono waveform, got shape {tuple(waveform.shape)}")
    window = torch.hann_window(FRAME_FFT, device=waveform.device, dtype=torch.float32)
    spectrum = torch.stft(
        waveform.float(),
        n_fft=FRAME_FFT,
        hop_length=FRAME_HOP,
        win_length=FRAME_FFT,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spectrum.real.square() + spectrum.imag.square()
    if power.shape[-1] < num_frames:
        power = torch.nn.functional.pad(power, (0, num_frames - power.shape[-1]))
    power = power[:, :num_frames]

    energy = power.sum(dim=0)
    log_power = torch.log(power + EPSILON)
    flux = (log_power[:, 1:] - log_power[:, :-1]).clamp_min(0.0).mean(dim=0)
    flux = torch.cat([flux[:1], flux], dim=0)

    bands = []
    for start, stop in _band_edges(power.shape[0]):
        bands.append(power[start:stop].sum(dim=0))
    band_stack = torch.stack(bands, dim=0)
    band_stack = band_stack / band_stack.sum(dim=0, keepdim=True).clamp_min(EPSILON)

    return torch.cat([energy[None], flux[None], band_stack], dim=0)


def intervals_to_frame_targets(
    intervals: list[tuple[float, float]],
    grid: FrameGrid,
) -> torch.Tensor:
    """Rasterise annotated event intervals onto the analysis grid."""

    target = torch.zeros(grid.num_frames, dtype=torch.float32)
    for onset, offset in intervals:
        start = max(0, min(grid.num_frames, grid.seconds_to_frame(onset)))
        stop = max(start, min(grid.num_frames, grid.seconds_to_frame(offset)))
        target[start:stop] = 1.0
    return target
