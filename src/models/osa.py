from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.models.common import DoubleConv, DownBlock, UpBlock, check_4d_or_5d_input, flatten_video_batch


class OrthogonalStateBlock(nn.Module):
    """Orthogonalized state-update-inspired feature block.

    The block learns a local state update and lightly suppresses correlated
    channel responses before adding it back to the residual stream. This keeps
    the implementation practical while approximating the intended OSA behavior.
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.update = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _decorrelate_channels(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.flatten(2)
        centered = flat - flat.mean(dim=-1, keepdim=True)
        normalized = F.normalize(centered, dim=-1, eps=1e-6)
        corr = torch.bmm(normalized, normalized.transpose(1, 2)) / max(1, h * w)
        eye = torch.eye(c, device=x.device, dtype=x.dtype).unsqueeze(0)
        corr = corr * (1.0 - eye)
        correction = torch.bmm(corr, flat) / max(1, c)
        return (flat - correction).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.update(self.norm(x))
        update = self._decorrelate_channels(update)
        return x + self.dropout(self.proj(update))


class AnatomicalPriorModule(nn.Module):
    """Feature-derived anatomical prior enhancement.

    The module predicts soft class-prior maps from features, builds class-wise
    context vectors, and feeds this context back to the feature map. It does not
    need manual priors or masks at inference.
    """

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.prior_logits = nn.Conv2d(channels, self.num_classes, kernel_size=1)
        self.enhance = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        priors = torch.softmax(self.prior_logits(x), dim=1)
        contexts = []
        for cls in range(self.num_classes):
            weight = priors[:, cls : cls + 1]
            denom = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            context = (x * weight).sum(dim=(2, 3), keepdim=True) / denom
            contexts.append(context * weight)
        class_context = torch.stack(contexts, dim=0).sum(dim=0) / max(1, self.num_classes)
        foreground_prior = priors[:, 1:].sum(dim=1, keepdim=True) if self.num_classes > 1 else priors[:, :1]
        enhanced = self.enhance(torch.cat([x, class_context, foreground_prior], dim=1))
        return x + self.scale * enhanced


class OSASegmentationModel(nn.Module):
    """OSA-style anatomical segmentation network.

    This is a research-inspired PyTorch approximation with an orthogonal state
    block and feature-derived anatomical prior module, not an official
    reproduction of a specific private implementation.
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

        self.enc1 = nn.Sequential(DoubleConv(self.in_channels, c, batch_norm=batch_norm, dropout=dropout), OrthogonalStateBlock(c, dropout=dropout))
        self.enc2 = nn.Sequential(DownBlock(c, c * 2, batch_norm=batch_norm, dropout=dropout), OrthogonalStateBlock(c * 2, dropout=dropout))
        self.enc3 = nn.Sequential(DownBlock(c * 2, c * 4, batch_norm=batch_norm, dropout=dropout), OrthogonalStateBlock(c * 4, dropout=dropout))
        self.enc4 = nn.Sequential(DownBlock(c * 4, c * 8, batch_norm=batch_norm, dropout=dropout), OrthogonalStateBlock(c * 8, dropout=dropout))
        self.bridge = nn.Sequential(
            DownBlock(c * 8, c * 16, batch_norm=batch_norm, dropout=dropout),
            OrthogonalStateBlock(c * 16, dropout=dropout),
            AnatomicalPriorModule(c * 16, self.num_classes),
        )
        self.up4 = UpBlock(c * 16, c * 8, c * 8, batch_norm=batch_norm, dropout=dropout)
        self.up3 = UpBlock(c * 8, c * 4, c * 4, batch_norm=batch_norm, dropout=dropout)
        self.up2 = UpBlock(c * 4, c * 2, c * 2, batch_norm=batch_norm, dropout=dropout)
        self.up1 = UpBlock(c * 2, c, c, batch_norm=batch_norm, dropout=dropout)
        self.head = nn.Conv2d(c, self.num_classes, kernel_size=1)

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bridge = self.bridge(e4)
        d4 = self.up4(bridge, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_4d_or_5d_input(x, self.in_channels, "OSA")
        if x.ndim == 4:
            return self._forward_4d(x)
        flat, batch_size, frames = flatten_video_batch(x)
        logits = self._forward_4d(flat)
        return logits.reshape(batch_size, frames, self.num_classes, *logits.shape[-2:]).mean(dim=1)
