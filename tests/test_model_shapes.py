import pytest
import torch

from src.model_registry import get_model


@pytest.mark.parametrize(
    "name,in_channels",
    [
        ("baseline_unet", 1),
        ("attention_unet", 1),
        ("unetpp", 1),
        ("multiresunet", 1),
        ("temporal_unet", 3),
    ],
)
def test_model_output_shapes(name, in_channels):
    model = get_model(
        name,
        in_channels=in_channels,
        num_classes=4,
        base_channels=4,
        dropout=0.0,
        temporal_window=in_channels,
    )
    model.eval()
    x = torch.randn(2, in_channels, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4, 64, 64)

