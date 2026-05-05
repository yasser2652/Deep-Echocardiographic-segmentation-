from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


_RESNET_SPECS = {
    "resnet18": {
        "builder": models.resnet18,
        "weights": models.ResNet18_Weights.DEFAULT,
        "channels": (64, 64, 128, 256, 512),
    },
    "resnet34": {
        "builder": models.resnet34,
        "weights": models.ResNet34_Weights.DEFAULT,
        "channels": (64, 64, 128, 256, 512),
    },
    "resnet50": {
        "builder": models.resnet50,
        "weights": models.ResNet50_Weights.DEFAULT,
        "channels": (64, 256, 512, 1024, 2048),
    },
}


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNet(nn.Module):
    """U-Net decoder on top of a torchvision ResNet encoder.

    Skip features come from the ResNet stem, layer1, layer2, and layer3. The
    layer4 feature map is the decoder bottleneck. For grayscale ultrasound
    input, the first convolution is replaced and initialized from the mean of
    pretrained RGB weights when ImageNet weights are used.
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        pretrained: bool = False,
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        if encoder_name not in _RESNET_SPECS:
            raise ValueError(f"Unsupported ResNet encoder '{encoder_name}'. Choose from {sorted(_RESNET_SPECS)}.")
        self.encoder_name = encoder_name
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        spec = _RESNET_SPECS[encoder_name]
        weights = spec["weights"] if pretrained else None
        try:
            encoder = spec["builder"](weights=weights)
        except Exception as exc:
            if not pretrained:
                raise
            warnings.warn(
                f"Could not load torchvision ImageNet weights for {encoder_name}: {exc}. "
                "Falling back to random encoder initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
            encoder = spec["builder"](weights=None)
        self._adapt_first_conv(encoder, self.in_channels)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        stem_ch, l1_ch, l2_ch, l3_ch, l4_ch = spec["channels"]
        d4 = max(base_channels * 8, 64)
        d3 = max(base_channels * 4, 64)
        d2 = max(base_channels * 2, 32)
        d1 = max(base_channels, 32)
        d0 = max(base_channels, 32)
        self.dec4 = DecoderBlock(l4_ch, l3_ch, d4, dropout=dropout)
        self.dec3 = DecoderBlock(d4, l2_ch, d3, dropout=dropout)
        self.dec2 = DecoderBlock(d3, l1_ch, d2, dropout=dropout)
        self.dec1 = DecoderBlock(d2, stem_ch, d1, dropout=dropout)
        self.final = nn.Sequential(
            nn.Conv2d(d1, d0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d0),
            nn.ReLU(inplace=True),
            nn.Conv2d(d0, self.num_classes, kernel_size=1),
        )

    @staticmethod
    def _adapt_first_conv(encoder: nn.Module, in_channels: int) -> None:
        old_conv = encoder.conv1
        if in_channels == old_conv.in_channels:
            return
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            elif in_channels < old_conv.in_channels:
                new_conv.weight.copy_(old_conv.weight[:, :in_channels])
            else:
                new_conv.weight[:, : old_conv.in_channels].copy_(old_conv.weight)
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                for ch in range(old_conv.in_channels, in_channels):
                    new_conv.weight[:, ch : ch + 1].copy_(mean_weight)
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        encoder.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"{self.__class__.__name__} expects [B, C, H, W], got {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")
        input_size = x.shape[-2:]
        stem = self.stem(x)
        l1 = self.layer1(self.maxpool(stem))
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        x = self.dec4(l4, l3)
        x = self.dec3(x, l2)
        x = self.dec2(x, l1)
        x = self.dec1(x, stem)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.final(x)


class ResNet18UNet(ResNetUNet):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_name="resnet18", **kwargs)


class ResNet34UNet(ResNetUNet):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_name="resnet34", **kwargs)


class ResNet50UNet(ResNetUNet):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_name="resnet50", **kwargs)
