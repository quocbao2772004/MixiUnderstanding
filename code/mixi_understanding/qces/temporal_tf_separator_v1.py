"""A compact temporal-span-conditioned complex-mask separator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels), nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class TemporalTFSeparatorV1(nn.Module):
    """Estimate a complex ratio mask from mixture TF features and a span gate."""

    def __init__(self, base_channels: int = 24, mask_scale: float = 2.0) -> None:
        super().__init__()
        c = int(base_channels)
        self.mask_scale = float(mask_scale)
        self.encoder0 = ConvBlock(4, c)
        self.down1 = nn.Sequential(nn.Conv2d(c, 2 * c, 4, stride=2, padding=1), nn.SiLU())
        self.encoder1 = ConvBlock(2 * c, 2 * c)
        self.down2 = nn.Sequential(nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1), nn.SiLU())
        self.encoder2 = ConvBlock(4 * c, 4 * c)
        self.down3 = nn.Sequential(nn.Conv2d(4 * c, 8 * c, 4, stride=2, padding=1), nn.SiLU())
        self.encoder3 = ConvBlock(8 * c, 8 * c)
        self.bottleneck = ConvBlock(8 * c, 8 * c)
        self.decoder2 = ConvBlock(8 * c + 4 * c, 4 * c)
        self.decoder1 = ConvBlock(4 * c + 2 * c, 2 * c)
        self.decoder0 = ConvBlock(2 * c + c, c)
        self.output = nn.Conv2d(c, 2, 1)
        # Start from mixture pass-through: complex mask = 1 + 0j.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _upsample(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        e0 = self.encoder0(features)
        e1 = self.encoder1(self.down1(e0))
        e2 = self.encoder2(self.down2(e1))
        e3 = self.encoder3(self.down3(e2))
        value = self.bottleneck(e3)
        value = self.decoder2(torch.cat((self._upsample(value, e2), e2), dim=1))
        value = self.decoder1(torch.cat((self._upsample(value, e1), e1), dim=1))
        value = self.decoder0(torch.cat((self._upsample(value, e0), e0), dim=1))
        delta = torch.tanh(self.output(value)) * self.mask_scale
        real = 1.0 + delta[:, 0]
        imaginary = delta[:, 1]
        # Complex-half is not supported by several STFT operators.  Keep the
        # convolutional trunk under autocast, then form a complex64 mask.
        return torch.complex(real.float(), imaginary.float())


def complex_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.complex(
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real,
    )
