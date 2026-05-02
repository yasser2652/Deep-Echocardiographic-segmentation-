from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm),
            ]
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

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
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, batch_norm=batch_norm, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
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
        self.enc1 = ConvBlock(in_channels, c, batch_norm=batch_norm, dropout=dropout)
        self.enc2 = ConvBlock(c, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.enc3 = ConvBlock(c * 2, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.enc4 = ConvBlock(c * 4, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(c * 8, c * 16, batch_norm=batch_norm, dropout=dropout)
        self.up4 = UpBlock(c * 16, c * 8, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.up3 = UpBlock(c * 8, c * 4, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.up2 = UpBlock(c * 4, c * 2, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.up1 = UpBlock(c * 2, c, c, batch_norm=batch_norm, dropout=dropout)
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

