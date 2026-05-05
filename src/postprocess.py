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


def keep_largest_component_per_class(binary: np.ndarray) -> np.ndarray:
    return keep_largest_component(binary)


def remove_small_objects(binary: np.ndarray, min_size: int = 64) -> np.ndarray:
    if min_size <= 0:
        return binary.astype(bool)
    return morphology.remove_small_objects(binary.astype(bool), min_size=min_size)


def fill_holes(binary: np.ndarray, area_threshold: int = 64) -> np.ndarray:
    out = ndimage.binary_fill_holes(binary.astype(bool))
    return morphology.remove_small_holes(out, area_threshold=max(4, int(area_threshold)))


def smooth_boundaries(binary: np.ndarray, radius: int = 2) -> np.ndarray:
    out = binary.astype(bool)
    if radius <= 0:
        return out
    out = morphology.binary_closing(out, morphology.disk(radius))
    out = morphology.binary_opening(out, morphology.disk(max(1, radius // 2)))
    return out


def postprocess_class_mask(
    binary: np.ndarray,
    keep_largest: bool = True,
    fill_holes_flag: bool | None = None,
    min_region_size: int = 64,
    smooth_boundaries_flag: bool | None = None,
    **kwargs,
) -> np.ndarray:
    if fill_holes_flag is None:
        fill_holes_flag = bool(kwargs.pop("fill_holes", True))
    if smooth_boundaries_flag is None:
        smooth_boundaries_flag = bool(kwargs.pop("smooth_boundaries", False))
    out = binary.astype(bool)
    if min_region_size > 0:
        out = remove_small_objects(out, min_size=min_region_size)
    if keep_largest:
        out = keep_largest_component_per_class(out)
    if fill_holes_flag:
        out = fill_holes(out, area_threshold=min_region_size)
    if smooth_boundaries_flag:
        out = smooth_boundaries(out)
    return out.astype(bool)


def enforce_no_overlap_priority(class_masks: dict[int, np.ndarray], shape: tuple[int, int], priority: list[int] | tuple[int, ...]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.int64)
    occupied = np.zeros(shape, dtype=bool)
    for cls in priority:
        if cls <= 0 or cls not in class_masks:
            continue
        processed = np.logical_and(class_masks[cls].astype(bool), ~occupied)
        out[processed] = int(cls)
        occupied |= processed
    return out


def postprocess_mask(mask: np.ndarray, num_classes: int = 4, config: dict | None = None) -> np.ndarray:
    config = config or {}
    priority = config.get("class_priority", config.get("priority", [1, 2, 3]))
    priority = [int(cls) for cls in priority if 0 < int(cls) < num_classes]
    for cls in range(1, num_classes):
        if cls not in priority:
            priority.append(cls)
    class_masks: dict[int, np.ndarray] = {}
    for cls in range(1, num_classes):
        binary = mask == cls
        processed = postprocess_class_mask(
            binary,
            keep_largest=bool(config.get("keep_largest_component_per_class", config.get("keep_largest_component", True))),
            fill_holes_flag=bool(config.get("fill_holes", True)),
            min_region_size=int(config.get("min_region_size", 64)),
            smooth_boundaries_flag=bool(config.get("smooth_boundaries", False)),
        )
        class_masks[cls] = processed
    if bool(config.get("enforce_no_overlap_priority", True)):
        return enforce_no_overlap_priority(class_masks, mask.shape, priority)
    out = np.zeros_like(mask, dtype=np.int64)
    for cls, processed in class_masks.items():
        out[processed] = cls
    return out
