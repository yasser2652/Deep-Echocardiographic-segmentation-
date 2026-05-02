from __future__ import annotations

import torch
from torch import nn

from src.models.unet import UNet


class TemporalAttentionFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class TemporalUNet(nn.Module):
    """Early-fusion temporal U-Net.

    Inputs are channel-stacked neighboring frames, e.g. [ED-1, ED, ED+1]. If the
    dataset cannot find neighbors it repeats the current frame, so this model is
    still safe to instantiate for smoke tests.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        batch_norm: bool = True,
        dropout: float = 0.0,
        temporal_window: int = 3,
        temporal_attention: bool = False,
    ) -> None:
        super().__init__()
        channels = max(in_channels, temporal_window)
        self.fusion = TemporalAttentionFusion(channels) if temporal_attention else nn.Identity()
        self.unet = UNet(
            in_channels=channels,
            num_classes=num_classes,
            base_channels=base_channels,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(self.fusion(x))

