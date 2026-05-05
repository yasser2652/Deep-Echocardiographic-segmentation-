from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import ndimage


def minmax_normalize(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lo = float(np.nanmin(image))
    hi = float(np.nanmax(image))
    if hi - lo < eps:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - lo) / (hi - lo)).astype(np.float32)


def zscore_normalize(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    mean = float(np.nanmean(image))
    std = float(np.nanstd(image))
    if std < eps:
        return np.zeros_like(image, dtype=np.float32)
    out = (image - mean) / std
    out = np.clip(out, -5.0, 5.0)
    return ((out + 5.0) / 10.0).astype(np.float32)


def ensure_channel_first(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image[None, ...]
    if image.ndim == 3:
        if image.shape[-1] in (1, 3, 4) and image.shape[0] not in (1, 3, 4):
            image = np.moveaxis(image, -1, 0)
        return image
    raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")


def _resize_2d(array: np.ndarray, size: tuple[int, int], order: int, mode: str) -> np.ndarray:
    array = np.asarray(array)
    if array.shape == size:
        return array.copy()
    zoom = (size[0] / array.shape[0], size[1] / array.shape[1])
    resized = ndimage.zoom(array, zoom=zoom, order=order, mode=mode)
    resized = resized[: size[0], : size[1]]
    pad_h = max(0, size[0] - resized.shape[0])
    pad_w = max(0, size[1] - resized.shape[1])
    if pad_h or pad_w:
        resized = np.pad(resized, ((0, pad_h), (0, pad_w)), mode="edge")
    return resized


def resize_image_mask(
    image: np.ndarray,
    mask: np.ndarray | None,
    image_size: int | tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = tuple(image_size)
    image = ensure_channel_first(image)
    resized_channels = [_resize_2d(channel, size, order=1, mode="reflect").astype(np.float32) for channel in image]
    out_image = np.stack(resized_channels, axis=0)
    out_mask = None
    if mask is not None:
        out_mask = _resize_2d(mask, size, order=0, mode="nearest").astype(np.int64)
    return out_image, out_mask


def _apply_to_channels(image: np.ndarray, func) -> np.ndarray:
    return np.stack([func(channel) for channel in image], axis=0).astype(np.float32)


def _random_scale_crop_pad(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    channels, height, width = image.shape
    new_h = max(8, int(round(height * scale)))
    new_w = max(8, int(round(width * scale)))
    img_scaled = np.stack(
        [_resize_2d(ch, (new_h, new_w), order=1, mode="reflect") for ch in image],
        axis=0,
    ).astype(np.float32)
    mask_scaled = _resize_2d(mask, (new_h, new_w), order=0, mode="nearest").astype(np.int64)
    if new_h >= height:
        top = int(rng.integers(0, new_h - height + 1))
        img_scaled = img_scaled[:, top : top + height, :]
        mask_scaled = mask_scaled[top : top + height, :]
    else:
        pad_top = (height - new_h) // 2
        pad_bottom = height - new_h - pad_top
        img_scaled = np.pad(img_scaled, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode="reflect")
        mask_scaled = np.pad(mask_scaled, ((pad_top, pad_bottom), (0, 0)), mode="edge")
    if new_w >= width:
        left = int(rng.integers(0, new_w - width + 1))
        img_scaled = img_scaled[:, :, left : left + width]
        mask_scaled = mask_scaled[:, left : left + width]
    else:
        pad_left = (width - new_w) // 2
        pad_right = width - new_w - pad_left
        img_scaled = np.pad(img_scaled, ((0, 0), (0, 0), (pad_left, pad_right)), mode="reflect")
        mask_scaled = np.pad(mask_scaled, ((0, 0), (pad_left, pad_right)), mode="edge")
    return img_scaled[:, :height, :width], mask_scaled[:height, :width]


def _parse_hw(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return value, value
    if isinstance(value, float):
        side = int(round(value))
        return side, side
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected crop size as int or [height, width], got {value!r}")


def random_crop_resize(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int | tuple[int, int] | list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = image.shape
    crop_h, crop_w = _parse_hw(crop_size, default=(height, width))
    crop_h = int(np.clip(crop_h, 1, height))
    crop_w = int(np.clip(crop_w, 1, width))
    if crop_h == height and crop_w == width:
        return image, mask
    top = int(rng.integers(0, height - crop_h + 1))
    left = int(rng.integers(0, width - crop_w + 1))
    image_crop = image[:, top : top + crop_h, left : left + crop_w]
    mask_crop = mask[top : top + crop_h, left : left + crop_w]
    image_resized = np.stack(
        [_resize_2d(ch, (height, width), order=1, mode="reflect") for ch in image_crop],
        axis=0,
    ).astype(np.float32)
    mask_resized = _resize_2d(mask_crop, (height, width), order=0, mode="nearest").astype(np.int64)
    return image_resized, mask_resized


def speckle_noise(image: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, std, size=image.shape).astype(np.float32)
    return np.clip(image + image * noise, 0.0, 1.0)


def acoustic_shadow(image: np.ndarray, rng: np.random.Generator, strength_range: tuple[float, float] = (0.25, 0.75)) -> np.ndarray:
    channels, height, width = image.shape
    shadow = np.ones((height, width), dtype=np.float32)
    center_x = float(rng.uniform(0, width))
    top_width = float(rng.uniform(width * 0.08, width * 0.25))
    bottom_width = float(rng.uniform(width * 0.18, width * 0.45))
    strength = float(rng.uniform(*strength_range))
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xx = np.arange(width, dtype=np.float32)[None, :]
    half_width = top_width * (1 - yy) + bottom_width * yy
    distance = np.abs(xx - center_x)
    wedge = distance <= half_width
    attenuation = 1.0 - strength * yy
    shadow[wedge] = np.broadcast_to(attenuation, (height, width))[wedge]
    return np.clip(image * shadow[None, ...], 0.0, 1.0).astype(np.float32)


def elastic_deform(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 12.0,
    sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = image.shape
    dx = ndimage.gaussian_filter((rng.random((height, width)) * 2 - 1), sigma, mode="reflect") * alpha
    dy = ndimage.gaussian_filter((rng.random((height, width)) * 2 - 1), sigma, mode="reflect") * alpha
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))
    deformed = []
    for channel in image:
        deformed.append(ndimage.map_coordinates(channel, indices, order=1, mode="reflect").reshape((height, width)))
    mask_deformed = ndimage.map_coordinates(mask, indices, order=0, mode="nearest").reshape((height, width))
    return np.stack(deformed, axis=0).astype(np.float32), mask_deformed.astype(np.int64)


def translate_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
    shift_yx: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    shift_y, shift_x = shift_yx
    shifted_image = _apply_to_channels(image, lambda ch: ndimage.shift(ch, shift=(shift_y, shift_x), order=1, mode="reflect"))
    shifted_mask = ndimage.shift(mask, shift=(shift_y, shift_x), order=0, mode="nearest").astype(np.int64)
    return shifted_image.astype(np.float32), shifted_mask


@dataclass
class SegmentationTransform:
    image_size: int | tuple[int, int] = 256
    training: bool = False
    augmentation: dict[str, Any] | None = None
    normalize: str = "minmax"
    z_score: bool = False

    def __post_init__(self) -> None:
        self.augmentation = self.augmentation or {}

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        image = ensure_channel_first(image).astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        normed = []
        for channel in image:
            if self.z_score or self.normalize == "zscore":
                normed.append(zscore_normalize(channel))
            else:
                normed.append(minmax_normalize(channel))
        image = np.stack(normed, axis=0).astype(np.float32)
        if mask is not None:
            mask = np.asarray(mask).astype(np.int64)

        image, mask = resize_image_mask(image, mask, self.image_size)
        if self.training and mask is not None and self.augmentation.get("enabled", True):
            image, mask = self._augment(image, mask)

        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        mask_tensor = None if mask is None else torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64)))
        return image_tensor, mask_tensor

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng()
        aug = self.augmentation
        if aug.get("horizontal_flip", False) and rng.random() < float(aug.get("horizontal_flip_prob", 0.5)):
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]

        degrees = float(aug.get("rotation_degrees", 0) or 0)
        if degrees > 0:
            angle = float(rng.uniform(-degrees, degrees))
            image = _apply_to_channels(
                image,
                lambda ch: ndimage.rotate(ch, angle, reshape=False, order=1, mode="reflect"),
            )
            mask = ndimage.rotate(mask, angle, reshape=False, order=0, mode="nearest").astype(np.int64)

        scale_range = aug.get("scale_range", None)
        if scale_range:
            scale = float(rng.uniform(float(scale_range[0]), float(scale_range[1])))
            image, mask = _random_scale_crop_pad(image, mask, scale, rng)

        translation_pixels = aug.get("translation_pixels", None)
        translation_fraction = float(aug.get("translation_fraction", 0) or 0)
        if translation_pixels is not None or translation_fraction > 0:
            _, height, width = image.shape
            if translation_pixels is None:
                max_y = height * translation_fraction
                max_x = width * translation_fraction
            elif isinstance(translation_pixels, (list, tuple)):
                max_y, max_x = float(translation_pixels[0]), float(translation_pixels[1])
            else:
                max_y = max_x = float(translation_pixels)
            shift = (float(rng.uniform(-max_y, max_y)), float(rng.uniform(-max_x, max_x)))
            image, mask = translate_image_mask(image, mask, shift)

        if rng.random() < float(aug.get("random_crop_prob", 0) or 0):
            crop_size = aug.get("random_crop_size", None)
            if crop_size is not None:
                image, mask = random_crop_resize(image, mask, crop_size, rng)

        brightness = float(aug.get("brightness", 0) or 0)
        contrast = float(aug.get("contrast", 0) or 0)
        if brightness > 0 or contrast > 0:
            b = float(rng.uniform(-brightness, brightness))
            c = float(rng.uniform(1 - contrast, 1 + contrast))
            image = np.clip((image - 0.5) * c + 0.5 + b, 0.0, 1.0)

        gain_range = aug.get("gain_range", None)
        if gain_range:
            gain = float(rng.uniform(float(gain_range[0]), float(gain_range[1])))
            image = np.clip(image * gain, 0.0, 1.0)

        gamma_range = aug.get("gamma_range", None)
        if gamma_range:
            gamma = float(rng.uniform(float(gamma_range[0]), float(gamma_range[1])))
            image = np.clip(image, 0.0, 1.0) ** gamma

        noise_std = float(aug.get("gaussian_noise_std", 0) or 0)
        if noise_std > 0:
            image = np.clip(image + rng.normal(0, noise_std, size=image.shape).astype(np.float32), 0.0, 1.0)

        speckle_std = float(aug.get("speckle_noise_std", 0) or 0)
        if speckle_std > 0:
            image = speckle_noise(image, speckle_std, rng)

        if rng.random() < float(aug.get("acoustic_shadow_prob", 0) or 0):
            strength = aug.get("acoustic_shadow_strength", (0.25, 0.75))
            image = acoustic_shadow(image, rng, strength_range=(float(strength[0]), float(strength[1])))

        if rng.random() < float(aug.get("elastic_prob", 0) or 0):
            image, mask = elastic_deform(
                image,
                mask,
                rng,
                alpha=float(aug.get("elastic_alpha", 12.0)),
                sigma=float(aug.get("elastic_sigma", 5.0)),
            )

        if rng.random() < float(aug.get("blur_prob", 0) or 0):
            sigma_range = aug.get("blur_sigma_range", (0.3, 1.2))
            sigma = float(rng.uniform(float(sigma_range[0]), float(sigma_range[1])))
            image = _apply_to_channels(image, lambda ch: ndimage.gaussian_filter(ch, sigma=sigma))

        return np.clip(image, 0.0, 1.0).astype(np.float32), mask.astype(np.int64)
