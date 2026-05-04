from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.common import DoubleConv, DownBlock, UpBlock, check_4d_or_5d_input, flatten_video_batch


class SpatialStateBlock(nn.Module):
    """Small vision-state-space-inspired spatial mixing block.

    This block is intentionally lightweight: it combines depthwise local
    filtering with normalized cumulative state summaries along height and width.
    It is inspired by vision Mamba/state-space modeling, but is not an official
    EchoVim implementation.
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _axis_state(x: torch.Tensor, dim: int) -> torch.Tensor:
        length = x.shape[dim]
        denom_shape = [1] * x.ndim
        denom_shape[dim] = length
        denom = torch.arange(1, length + 1, device=x.device, dtype=x.dtype).reshape(denom_shape)
        forward = torch.cumsum(x, dim=dim) / denom
        backward = torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim) / denom, dims=[dim])
        return 0.5 * (forward + backward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        state, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        state = self.dwconv(state)
        mixed = 0.5 * (self._axis_state(state, dim=2) + self._axis_state(state, dim=3))
        mixed = mixed * torch.sigmoid(gate)
        return residual + self.dropout(self.out_proj(mixed))


class EchoVimSegmentationModel(nn.Module):
    """EchoVim-style segmentation model for echocardiography frames.

    This is a repo-compatible approximation using convolutional patch/stem
    layers and lightweight state-space-inspired spatial blocks. Video input is
    supported by processing frames independently and averaging frame logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        batch_norm: bool = True,
        dropout: float = 0.0,
        input_size: tuple[int, int] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.input_size = input_size
        c = int(base_channels)

        self.stem = DoubleConv(self.in_channels, c, batch_norm=batch_norm, dropout=dropout)
        self.vim1 = SpatialStateBlock(c, dropout=dropout)
        self.down1 = DownBlock(c, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.vim2 = SpatialStateBlock(c * 2, dropout=dropout)
        self.down2 = DownBlock(c * 2, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.vim3 = SpatialStateBlock(c * 4, dropout=dropout)
        self.down3 = DownBlock(c * 4, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.bridge = nn.Sequential(DoubleConv(c * 8, c * 16, batch_norm=batch_norm, dropout=dropout), SpatialStateBlock(c * 16, dropout=dropout))

        self.up3 = UpBlock(c * 16, c * 8, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.up2 = UpBlock(c * 8, c * 4, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.up1 = UpBlock(c * 4, c * 2, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.up0 = UpBlock(c * 2, c, c, batch_norm=batch_norm, dropout=dropout)
        self.head = nn.Conv2d(c, self.num_classes, kernel_size=1)

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.vim1(self.stem(x))
        e2 = self.vim2(self.down1(e1))
        e3 = self.vim3(self.down2(e2))
        e4 = self.down3(e3)
        bridge = self.bridge(e4)
        d3 = self.up3(bridge, e4)
        d2 = self.up2(d3, e3)
        d1 = self.up1(d2, e2)
        d0 = self.up0(d1, e1)
        return self.head(d0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_4d_or_5d_input(x, self.in_channels, "EchoVim")
        if x.ndim == 4:
            return self._forward_4d(x)
        flat, batch_size, frames = flatten_video_batch(x)
        logits = self._forward_4d(flat)
        return logits.reshape(batch_size, frames, self.num_classes, *logits.shape[-2:]).mean(dim=1)
