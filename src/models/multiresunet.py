from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, alpha: float = 1.67) -> None:
        super().__init__()
        filters = max(3, int(out_channels * alpha))
        f1 = max(1, filters // 6)
        f2 = max(1, filters // 3)
        f3 = max(1, filters - f1 - f2)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, f1 + f2 + f3, kernel_size=1, bias=False),
            nn.BatchNorm2d(f1 + f2 + f3),
        )
        self.conv3 = ConvBNReLU(in_channels, f1)
        self.conv5 = ConvBNReLU(f1, f2)
        self.conv7 = ConvBNReLU(f2, f3)
        self.bn = nn.BatchNorm2d(f1 + f2 + f3)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = f1 + f2 + f3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = self.conv3(x)
        x5 = self.conv5(x3)
        x7 = self.conv7(x5)
        out = torch.cat([x3, x5, x7], dim=1)
        return self.relu(self.bn(out + self.shortcut(x)))


class ResPath(nn.Module):
    def __init__(self, channels: int, length: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "conv": ConvBNReLU(channels, channels),
                        "skip": nn.Sequential(
                            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                            nn.BatchNorm2d(channels),
                        ),
                    }
                )
                for _ in range(length)
            ]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = self.relu(block["conv"](x) + block["skip"](x))
        return x


class MultiResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del batch_norm, dropout
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.mres1 = MultiResBlock(in_channels, c)
        self.res1 = ResPath(self.mres1.out_channels, 4)
        self.mres2 = MultiResBlock(self.mres1.out_channels, c * 2)
        self.res2 = ResPath(self.mres2.out_channels, 3)
        self.mres3 = MultiResBlock(self.mres2.out_channels, c * 4)
        self.res3 = ResPath(self.mres3.out_channels, 2)
        self.mres4 = MultiResBlock(self.mres3.out_channels, c * 8)
        self.res4 = ResPath(self.mres4.out_channels, 1)
        self.bridge = MultiResBlock(self.mres4.out_channels, c * 16)

        self.up4 = nn.ConvTranspose2d(self.bridge.out_channels, self.mres4.out_channels, kernel_size=2, stride=2)
        self.dec4 = MultiResBlock(self.mres4.out_channels * 2, c * 8)
        self.up3 = nn.ConvTranspose2d(self.dec4.out_channels, self.mres3.out_channels, kernel_size=2, stride=2)
        self.dec3 = MultiResBlock(self.mres3.out_channels * 2, c * 4)
        self.up2 = nn.ConvTranspose2d(self.dec3.out_channels, self.mres2.out_channels, kernel_size=2, stride=2)
        self.dec2 = MultiResBlock(self.mres2.out_channels * 2, c * 2)
        self.up1 = nn.ConvTranspose2d(self.dec2.out_channels, self.mres1.out_channels, kernel_size=2, stride=2)
        self.dec1 = MultiResBlock(self.mres1.out_channels * 2, c)
        self.head = nn.Conv2d(self.dec1.out_channels, num_classes, kernel_size=1)

    def _upcat(self, x: torch.Tensor, skip: torch.Tensor, up: nn.Module) -> torch.Tensor:
        x = up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1_raw = self.mres1(x)
        e1 = self.res1(e1_raw)
        e2_raw = self.mres2(self.pool(e1_raw))
        e2 = self.res2(e2_raw)
        e3_raw = self.mres3(self.pool(e2_raw))
        e3 = self.res3(e3_raw)
        e4_raw = self.mres4(self.pool(e3_raw))
        e4 = self.res4(e4_raw)
        b = self.bridge(self.pool(e4_raw))
        d4 = self.dec4(self._upcat(b, e4, self.up4))
        d3 = self.dec3(self._upcat(d4, e3, self.up3))
        d2 = self.dec2(self._upcat(d3, e2, self.up2))
        d1 = self.dec1(self._upcat(d2, e1, self.up1))
        return self.head(d1)

