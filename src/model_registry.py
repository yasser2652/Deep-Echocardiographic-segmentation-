from __future__ import annotations

from typing import Any

from torch import nn

from src.models import AttentionUNet, MultiResUNet, TemporalUNet, UNet, UNetPlusPlus


MODEL_REGISTRY = {
    "baseline_unet": UNet,
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "unetpp": UNetPlusPlus,
    "unet++": UNetPlusPlus,
    "multiresunet": MultiResUNet,
    "temporal_unet": TemporalUNet,
}


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def get_model(
    name: str,
    in_channels: int = 1,
    num_classes: int = 4,
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
    return get_model(
        model_name,
        in_channels=in_channels,
        num_classes=int(config.get("num_classes", 4)),
        base_channels=int(params.get("base_channels", 32)),
        batch_norm=bool(params.get("batch_norm", True)),
        dropout=float(params.get("dropout", 0.0)),
        temporal_window=temporal_window,
        temporal_attention=bool(config.get("temporal_attention", False)),
    )

