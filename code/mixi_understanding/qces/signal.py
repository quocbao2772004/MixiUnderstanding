"""Signal operations with deterministic CUDA-compatible boundary handling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def deterministic_reflect_pad_1d(
    tensor: torch.Tensor, left: int, right: int
) -> torch.Tensor:
    """Match one-dimensional reflection padding without CUDA atomic adds."""

    if tensor.ndim < 1 or left < 0 or right < 0:
        raise ValueError("invalid deterministic reflection padding request")
    length = tensor.size(-1)
    if left >= length or right >= length:
        raise ValueError(
            "reflection padding must be smaller than the input length"
        )
    pieces = []
    if left:
        pieces.append(tensor[..., 1 : left + 1].flip(-1))
    pieces.append(tensor)
    if right:
        pieces.append(tensor[..., -right - 1 : -1].flip(-1))
    return torch.cat(pieces, dim=-1)


def qces_stft(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    center: bool = True,
    pad_mode: str = "reflect",
    return_complex: bool = True,
) -> torch.Tensor:
    """Run STFT exactly, replacing only nondeterministic CUDA reflect padding."""

    if (
        center
        and pad_mode == "reflect"
        and torch.are_deterministic_algorithms_enabled()
    ):
        padding = n_fft // 2
        waveform = deterministic_reflect_pad_1d(
            waveform, padding, padding
        )
        center = False
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        return_complex=return_complex,
    )


def qces_linear_interpolate_1d(
    tensor: torch.Tensor, size: int
) -> torch.Tensor:
    """Linear 1-D resize with deterministic CUDA backward when requested."""

    if tensor.ndim != 3:
        raise ValueError("linear interpolation expects shape [B, C, T]")
    if size <= 0:
        raise ValueError("linear interpolation size must be positive")
    if not torch.are_deterministic_algorithms_enabled():
        return F.interpolate(
            tensor, size=size, mode="linear", align_corners=False
        )
    input_size = tensor.size(-1)
    if input_size == size:
        return tensor
    if input_size == 1:
        return tensor.expand(*tensor.shape[:-1], size)
    # This is PyTorch's align_corners=False source-coordinate transform.
    coordinates = (
        (torch.arange(size, device=tensor.device, dtype=tensor.dtype) + 0.5)
        * (input_size / size)
        - 0.5
    ).clamp_(0.0, float(input_size - 1))
    left = coordinates.floor().to(torch.long)
    right = (left + 1).clamp_max(input_size - 1)
    right_weight = (coordinates - left).to(dtype=tensor.dtype)
    left_values = tensor.index_select(-1, left)
    right_values = tensor.index_select(-1, right)
    return left_values + (right_values - left_values) * right_weight


def qces_smooth_1d(
    tensor: torch.Tensor,
    kernel: tuple[float, ...] = (1.0, 2.0, 3.0, 2.0, 1.0),
) -> torch.Tensor:
    """Smooth ``[B, C, T]`` features with deterministic edge replication.

    The implementation is a weighted sum of index selections instead of a
    padded convolution.  This keeps both the CPU and strict CUDA backward
    paths deterministic, including at the boundary frames.
    """

    if tensor.ndim != 3:
        raise ValueError("temporal smoothing expects shape [B, C, T]")
    if not kernel or len(kernel) % 2 != 1:
        raise ValueError("temporal smoothing needs a non-empty odd kernel")
    if any(weight < 0 for weight in kernel) or sum(kernel) <= 0:
        raise ValueError("temporal smoothing weights must be non-negative")
    frame_count = tensor.size(-1)
    if frame_count <= 0:
        raise ValueError("temporal smoothing needs at least one frame")
    center = len(kernel) // 2
    base = torch.arange(frame_count, device=tensor.device)
    total = tensor.new_zeros(tensor.shape)
    normalization = float(sum(kernel))
    for offset, weight in enumerate(kernel):
        if weight == 0:
            continue
        indices = (base + offset - center).clamp(0, frame_count - 1)
        total = total + tensor.index_select(-1, indices) * (weight / normalization)
    return total
