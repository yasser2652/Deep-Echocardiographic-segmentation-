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
        ("gdkvm", 1),
        ("echovim", 1),
        ("osa", 1),
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


@pytest.mark.parametrize("name", ["gdkvm", "echovim", "osa"])
def test_advanced_models_video_input_shape(name):
    model = get_model(name, in_channels=1, num_classes=4, base_channels=4, dropout=0.0)
    model.eval()
    x = torch.randn(2, 3, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4, 64, 64)


@pytest.mark.parametrize("name", ["gdkvm", "echovim", "osa"])
def test_advanced_models_training_step(name):
    model = get_model(name, in_channels=1, num_classes=4, base_channels=2, dropout=0.0)
    model.train()
    x = torch.randn(2, 1, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))
    loss = torch.nn.functional.cross_entropy(model(x), target)
    loss.backward()
    assert torch.isfinite(loss)
