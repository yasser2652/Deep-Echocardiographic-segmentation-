from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.models.common import (
    DoubleConv,
    DownBlock,
    UpBlock,
    center_frame_index,
    check_4d_or_5d_input,
    flatten_video_batch,
)


class DynamicKernelValueMemory(nn.Module):
    """Lightweight dynamic key-value memory block.

    This is a repo-compatible, research-inspired approximation of GDKVM for
    dense segmentation. It uses query/key/value projections to build a compact
    channel memory from the current feature map, then writes the retrieved
    context back as a residual spatial feature.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.scale = math.sqrt(max(1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).flatten(2) / self.scale
        k = self.key(x).flatten(2)
        v = self.value(x).flatten(2)
        q = torch.softmax(q, dim=-1)
        k = torch.softmax(k, dim=-1)
        memory = torch.bmm(v, k.transpose(1, 2))
        retrieved = torch.bmm(memory, q).reshape(b, c, h, w)
        return x + self.out(retrieved)


class GDKVMSegmentationModel(nn.Module):
    """Practical GDKVM-style echocardiography segmentation network.

    The model is not an official reproduction. It keeps the U-Net-like
    encoder/decoder shape that this repository already trains well, and adds a
    dynamic key-value memory bottleneck plus optional temporal fusion for
    [B, T, C, H, W] cine inputs.
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

        self.enc1 = DoubleConv(self.in_channels, c, batch_norm=batch_norm, dropout=dropout)
        self.enc2 = DownBlock(c, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.enc3 = DownBlock(c * 2, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.enc4 = DownBlock(c * 4, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.bridge = DownBlock(c * 8, c * 16, batch_norm=batch_norm, dropout=dropout)
        self.memory = DynamicKernelValueMemory(c * 16)
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(c * 32, c * 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(c * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 16, c * 16, kernel_size=1),
        )

        self.up4 = UpBlock(c * 16, c * 8, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.up3 = UpBlock(c * 8, c * 4, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.up2 = UpBlock(c * 4, c * 2, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.up1 = UpBlock(c * 2, c, c, batch_norm=batch_norm, dropout=dropout)
        self.head = nn.Conv2d(c, self.num_classes, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bridge(e4)
        return e1, e2, e3, e4, b

    def _decode(
        self,
        bridge: torch.Tensor,
        e4: torch.Tensor,
        e3: torch.Tensor,
        e2: torch.Tensor,
        e1: torch.Tensor,
    ) -> torch.Tensor:
        d4 = self.up4(bridge, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)

    @staticmethod
    def _select_center(features: torch.Tensor, batch_size: int, frames: int) -> torch.Tensor:
        center = center_frame_index(frames)
        return features.reshape(batch_size, frames, *features.shape[1:])[:, center]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_4d_or_5d_input(x, self.in_channels, "GDKVM")
        if x.ndim == 4:
            e1, e2, e3, e4, bridge = self._encode(x)
            return self._decode(self.memory(bridge), e4, e3, e2, e1)

        flat, batch_size, frames = flatten_video_batch(x)
        e1, e2, e3, e4, bridge = self._encode(flat)
        bridge_seq = bridge.reshape(batch_size, frames, *bridge.shape[1:])
        center_bridge = bridge_seq[:, center_frame_index(frames)]
        temporal_context = bridge_seq.mean(dim=1)
        bridge = self.memory(center_bridge)
        bridge = bridge + self.temporal_fusion(torch.cat([bridge, temporal_context], dim=1))
        return self._decode(
            bridge,
            self._select_center(e4, batch_size, frames),
            self._select_center(e3, batch_size, frames),
            self._select_center(e2, batch_size, frames),
            self._select_center(e1, batch_size, frames),
        )
