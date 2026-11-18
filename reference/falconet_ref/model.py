"""CenterNet-lite: a small anchor-free detector for tiny objects.

Input: grayscale images, values in [0, 1], shape [B, 1, H, W] with H and W
divisible by 32 (Anti-UAV410 frames are 640 x 512).

Output, on a grid 4x coarser than the input ([B, C, H/4, W/4]):
- heatmap logits (C=1): where object centres are,
- log size (C=2): log of box width and height in pixels, at the centre cell,
- offset logits (C=2): sub-cell position of the centre (sigmoid -> [0, 1]).

Architecture:

    stem   1 -> 32  stride 4   (two stride-2 convs + residual block)
    c8    32 -> 64  stride 8
    c16   64 -> 128 stride 16  (two residual blocks)
    c32  128 -> 128 stride 32  (receptive field for the largest drones)
    FPN-lite top-down: 1x1 laterals to 64 channels, nearest x2 upsample + add,
    back to stride 4, then a 3x3 conv
    heads: 3x3 conv + ReLU + 1x1 conv each

Stride 4 matters: half of Anti-UAV410's boxes are smaller than 32 x 32 px,
which at stride 16 would be a single cell.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# sigmoid(-2.19) ~ 0.1: start from "almost no object anywhere" (focal-loss prior).
HEATMAP_PRIOR_BIAS = -2.19
STRIDE = 4


class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, relu: bool = True) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class Residual(nn.Module):
    """Two 3x3 conv-BN layers with an identity shortcut (ResNet basic block)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(ConvBNReLU(channels, channels), ConvBNReLU(channels, channels, relu=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x))


def head(c_in: int, c_out: int, bias: float | None = None) -> nn.Sequential:
    out = nn.Conv2d(64, c_out, 1)
    if bias is not None:
        nn.init.constant_(out.bias, bias)
    return nn.Sequential(nn.Conv2d(c_in, 64, 3, padding=1), nn.ReLU(inplace=True), out)


class CenterNetLite(nn.Module):
    def __init__(self, neck: int = 64, pixel_mean: float = 0.5, pixel_std: float = 0.25) -> None:
        super().__init__()
        # Normalisation lives inside the model, so Rust feeds raw [0, 1] pixels.
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean))
        self.register_buffer("pixel_std", torch.tensor(pixel_std))

        self.stem = nn.Sequential(ConvBNReLU(1, 16, stride=2), ConvBNReLU(16, 32, stride=2), Residual(32))
        self.c8 = nn.Sequential(ConvBNReLU(32, 64, stride=2), Residual(64))
        self.c16 = nn.Sequential(ConvBNReLU(64, 128, stride=2), Residual(128), Residual(128))
        self.c32 = nn.Sequential(ConvBNReLU(128, 128, stride=2), Residual(128))

        self.lat4 = nn.Conv2d(32, neck, 1)
        self.lat8 = nn.Conv2d(64, neck, 1)
        self.lat16 = nn.Conv2d(128, neck, 1)
        self.lat32 = nn.Conv2d(128, neck, 1)
        self.smooth = ConvBNReLU(neck, neck)

        self.heatmap = head(neck, 1, bias=HEATMAP_PRIOR_BIAS)
        self.log_size = head(neck, 2)
        self.offset = head(neck, 2)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = (image - self.pixel_mean) / self.pixel_std
        c4 = self.stem(x)
        c8 = self.c8(c4)
        c16 = self.c16(c8)
        c32 = self.c32(c16)

        up = lambda t: F.interpolate(t, scale_factor=2.0, mode="nearest")  # noqa: E731
        p16 = self.lat16(c16) + up(self.lat32(c32))
        p8 = self.lat8(c8) + up(p16)
        p4 = self.smooth(self.lat4(c4) + up(p8))
        return self.heatmap(p4), self.log_size(p4), self.offset(p4)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
