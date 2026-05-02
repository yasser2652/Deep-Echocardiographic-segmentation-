from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.unet import ConvBlock


def upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class UNetPlusPlus(nn.Module):
    """Readable U-Net++ with four encoder depths and nested decoder skips."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        batch_norm: bool = True,
        dropout: float = 0.0,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.x00 = ConvBlock(in_channels, c, batch_norm=batch_norm, dropout=dropout)
        self.x10 = ConvBlock(c, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.x20 = ConvBlock(c * 2, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.x30 = ConvBlock(c * 4, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.x40 = ConvBlock(c * 8, c * 16, batch_norm=batch_norm, dropout=dropout)

        self.x01 = ConvBlock(c + c * 2, c, batch_norm=batch_norm, dropout=dropout)
        self.x11 = ConvBlock(c * 2 + c * 4, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.x21 = ConvBlock(c * 4 + c * 8, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.x31 = ConvBlock(c * 8 + c * 16, c * 8, batch_norm=batch_norm, dropout=dropout)

        self.x02 = ConvBlock(c * 2 + c * 2, c, batch_norm=batch_norm, dropout=dropout)
        self.x12 = ConvBlock(c * 2 * 2 + c * 4, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.x22 = ConvBlock(c * 4 * 2 + c * 8, c * 4, batch_norm=batch_norm, dropout=dropout)

        self.x03 = ConvBlock(c * 3 + c * 2, c, batch_norm=batch_norm, dropout=dropout)
        self.x13 = ConvBlock(c * 2 * 3 + c * 4, c * 2, batch_norm=batch_norm, dropout=dropout)

        self.x04 = ConvBlock(c * 4 + c * 2, c, batch_norm=batch_norm, dropout=dropout)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        x01 = self.x01(torch.cat([x00, upsample_like(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, upsample_like(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, upsample_like(x30, x20)], dim=1))
        x31 = self.x31(torch.cat([x30, upsample_like(x40, x30)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, upsample_like(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, upsample_like(x21, x10)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, upsample_like(x31, x20)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, upsample_like(x12, x00)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, upsample_like(x22, x10)], dim=1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, upsample_like(x13, x00)], dim=1))
        return self.head(x04)

