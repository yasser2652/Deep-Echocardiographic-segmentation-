from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.unet import ConvBlock


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.relu(self.gate_proj(gate) + self.skip_proj(skip)))
        return skip * attention


class AttentionUpBlock(nn.Module):
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
        self.attention = AttentionGate(out_channels, skip_channels, max(1, out_channels // 2))
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, batch_norm=batch_norm, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attention(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, c, batch_norm=batch_norm, dropout=dropout)
        self.enc2 = ConvBlock(c, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.enc3 = ConvBlock(c * 2, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.enc4 = ConvBlock(c * 4, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.bridge = ConvBlock(c * 8, c * 16, batch_norm=batch_norm, dropout=dropout)
        self.up4 = AttentionUpBlock(c * 16, c * 8, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.up3 = AttentionUpBlock(c * 8, c * 4, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.up2 = AttentionUpBlock(c * 4, c * 2, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.up1 = AttentionUpBlock(c * 2, c, c, batch_norm=batch_norm, dropout=dropout)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bridge(self.pool(e4))
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)

