from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def check_4d_or_5d_input(x: torch.Tensor, in_channels: int, model_name: str) -> None:
    if x.ndim not in (4, 5):
        raise ValueError(
            f"{model_name} expects [B, C, H, W] or [B, T, C, H, W] input, "
            f"but received shape {tuple(x.shape)}."
        )
    channel_dim = 1 if x.ndim == 4 else 2
    if x.shape[channel_dim] != in_channels:
        raise ValueError(
            f"{model_name} was created for in_channels={in_channels}, "
            f"but received {x.shape[channel_dim]} channels in shape {tuple(x.shape)}."
        )


def flatten_video_batch(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if x.ndim != 5:
        raise ValueError(f"Expected video input [B, T, C, H, W], got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    return x.reshape(b * t, c, h, w), b, t


def center_frame_index(num_frames: int) -> int:
    return max(0, int(num_frames) // 2)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        batch_norm: bool = True,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            ConvNormAct(in_channels, out_channels, batch_norm=batch_norm),
            ConvNormAct(out_channels, out_channels, batch_norm=batch_norm),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, batch_norm=batch_norm, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, batch_norm=batch_norm, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def as_2tuple(value: int | Tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected input_size/image_size with two values, got {value}")
    return int(value[0]), int(value[1])
