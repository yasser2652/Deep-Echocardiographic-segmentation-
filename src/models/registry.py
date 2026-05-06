from __future__ import annotations

from typing import Any

from torch import nn

from src.models.attention_unet import AttentionUNet
from src.models.echovim import EchoVimSegmentationModel
from src.models.gdkvm import GDKVMSegmentationModel
from src.models.multiresunet import MultiResUNet
from src.models.osa import OSASegmentationModel
from src.models.resnet_unet import ResNet18UNet, ResNet34UNet, ResNet50UNet
from src.models.temporal_unet import TemporalUNet
from src.models.unet import UNet
from src.models.unetpp import UNetPlusPlus


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline_unet": UNet,
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "unetpp": UNetPlusPlus,
    "unet++": UNetPlusPlus,
    "multiresunet": MultiResUNet,
    "temporal_unet": TemporalUNet,
    "gdkvm": GDKVMSegmentationModel,
    "echovim": EchoVimSegmentationModel,
    "echo_vim": EchoVimSegmentationModel,
    "osa": OSASegmentationModel,
    "resnet18_unet": ResNet18UNet,
    "resnet34_unet": ResNet34UNet,
    "resnet50_unet": ResNet50UNet,
}

ADVANCED_MODEL_KEYS = {"gdkvm", "echovim", "echo_vim", "osa"}
RESNET_UNET_KEYS = {"resnet18_unet", "resnet34_unet", "resnet50_unet"}


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def get_model(
    name: str,
    in_channels: int = 1,
    num_classes: int = 4,
    input_size: tuple[int, int] | None = None,
    base_channels: int = 32,
    batch_norm: bool = True,
    dropout: float = 0.0,
    **kwargs: Any,
) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {', '.join(available_models())}")
    model_cls = MODEL_REGISTRY[key]
    if key == "temporal_unet":
        temporal_window = int(kwargs.get("temporal_window", max(1, in_channels)))
        in_channels = max(in_channels, temporal_window)
        return model_cls(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            batch_norm=batch_norm,
            dropout=dropout,
            temporal_window=temporal_window,
            temporal_attention=bool(kwargs.get("temporal_attention", False)),
        )
    if key in ADVANCED_MODEL_KEYS | RESNET_UNET_KEYS:
        extra_kwargs = dict(kwargs)
        pretrained = bool(extra_kwargs.pop("pretrained", extra_kwargs.pop("imagenet_pretrained", False)))
        return model_cls(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            batch_norm=batch_norm,
            dropout=dropout,
            input_size=input_size,
            pretrained=pretrained,
            **extra_kwargs,
        )
    return model_cls(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        batch_norm=batch_norm,
        dropout=dropout,
    )


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    params = dict(config.get("model_params", {}) or {})
    model_name = str(config.get("model", "baseline_unet"))
    temporal_window = int(config.get("temporal_window", 1))
    in_channels = int(config.get("in_channels", 1))
    if model_name.lower() == "temporal_unet" or temporal_window > 1:
        in_channels = max(in_channels, temporal_window)
    input_size = config.get("input_size")
    if input_size is None:
        image_size = config.get("image_size")
        if isinstance(image_size, int):
            input_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            input_size = (int(image_size[0]), int(image_size[1]))
    return get_model(
        model_name,
        in_channels=in_channels,
        num_classes=int(config.get("num_classes", 4)),
        input_size=input_size,
        base_channels=int(params.get("base_channels", 32)),
        batch_norm=bool(params.get("batch_norm", True)),
        dropout=float(params.get("dropout", 0.0)),
        pretrained=bool(params.get("pretrained", params.get("imagenet_pretrained", config.get("pretrained", False)))),
        temporal_window=temporal_window,
        temporal_attention=bool(config.get("temporal_attention", False)),
    )
