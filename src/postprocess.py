from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure, morphology


def keep_largest_component(binary: np.ndarray) -> np.ndarray:
    labels = measure.label(binary.astype(bool), connectivity=2)
    if labels.max() == 0:
        return binary.astype(bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return labels == largest


def postprocess_class_mask(
    binary: np.ndarray,
    keep_largest: bool = True,
    fill_holes: bool = True,
    min_region_size: int = 64,
    smooth_boundaries: bool = False,
) -> np.ndarray:
    out = binary.astype(bool)
    if min_region_size > 0:
        out = morphology.remove_small_objects(out, min_size=min_region_size)
    if keep_largest:
        out = keep_largest_component(out)
    if fill_holes:
        out = ndimage.binary_fill_holes(out)
        out = morphology.remove_small_holes(out, area_threshold=max(4, min_region_size))
    if smooth_boundaries:
        out = morphology.binary_closing(out, morphology.disk(2))
        out = morphology.binary_opening(out, morphology.disk(1))
    return out.astype(bool)


def postprocess_mask(mask: np.ndarray, num_classes: int = 4, config: dict | None = None) -> np.ndarray:
    config = config or {}
    out = np.zeros_like(mask, dtype=np.int64)
    occupied = np.zeros_like(mask, dtype=bool)
    for cls in range(1, num_classes):
        binary = mask == cls
        processed = postprocess_class_mask(
            binary,
            keep_largest=bool(config.get("keep_largest_component", True)),
            fill_holes=bool(config.get("fill_holes", True)),
            min_region_size=int(config.get("min_region_size", 64)),
            smooth_boundaries=bool(config.get("smooth_boundaries", False)),
        )
        processed = np.logical_and(processed, ~occupied)
        out[processed] = cls
        occupied |= processed
    return out

