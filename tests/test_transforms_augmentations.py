import numpy as np
import pytest
import yaml

from src.transforms import SegmentationTransform


def sample_image_mask(size: int = 64):
    yy, xx = np.mgrid[0:size, 0:size]
    image = (xx + yy).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.int64)
    mask[((yy - size // 2) ** 2) / 14**2 + ((xx - size // 2) ** 2) / 10**2 <= 1] = 1
    mask[((yy - size // 3) ** 2) / 7**2 + ((xx - size // 2) ** 2) / 12**2 <= 1] = 3
    return image, mask


@pytest.mark.parametrize(
    "augmentation",
    [
        {"rotation_degrees": 12},
        {"scale_range": [0.85, 1.15]},
        {"translation_fraction": 0.05},
        {"translation_pixels": [3, 4]},
        {"random_crop_prob": 1.0, "random_crop_size": [40, 40]},
        {"horizontal_flip": True, "horizontal_flip_prob": 1.0},
        {"brightness": 0.15, "contrast": 0.15},
        {"gaussian_noise_std": 0.03},
        {"speckle_noise_std": 0.08},
        {"gain_range": [1.15, 1.15]},
        {"acoustic_shadow_prob": 1.0, "acoustic_shadow_strength": [0.5, 0.5]},
        {"elastic_prob": 1.0, "elastic_alpha": 2.0, "elastic_sigma": 4.0},
        {"blur_prob": 1.0, "blur_sigma_range": [0.8, 0.8]},
        {"gamma_range": [1.25, 1.25]},
    ],
)
def test_each_listed_augmentation_runs_and_preserves_contract(augmentation):
    image, mask = sample_image_mask()
    transform = SegmentationTransform(
        image_size=64,
        training=True,
        augmentation={"enabled": True, **augmentation},
    )
    image_tensor, mask_tensor = transform(image, mask)
    assert image_tensor.shape == (1, 64, 64)
    assert mask_tensor.shape == (64, 64)
    assert image_tensor.dtype.is_floating_point
    assert mask_tensor.dtype.is_floating_point is False
    assert float(image_tensor.min()) >= 0.0
    assert float(image_tensor.max()) <= 1.0
    assert set(np.unique(mask_tensor.numpy())).issubset({0, 1, 2, 3})


def test_prompt_augmentations_are_configurable_from_config_yaml():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    augmentation = config["augmentation"]
    required_keys = {
        "horizontal_flip",
        "horizontal_flip_prob",
        "rotation_degrees",
        "scale_range",
        "translation_fraction",
        "translation_pixels",
        "random_crop_prob",
        "random_crop_size",
        "brightness",
        "contrast",
        "gaussian_noise_std",
        "speckle_noise_std",
        "gain_range",
        "acoustic_shadow_prob",
        "acoustic_shadow_strength",
        "elastic_prob",
        "elastic_alpha",
        "elastic_sigma",
        "blur_prob",
        "blur_sigma_range",
        "gamma_range",
    }
    assert required_keys.issubset(augmentation)
