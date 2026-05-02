from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing: tuple[float, float] | None = None) -> np.ndarray:
    spacing = spacing or (1.0, 1.0)
    surf_a = _surface(mask_a)
    surf_b = _surface(mask_b)
    if not surf_a.any() and not surf_b.any():
        return np.array([0.0], dtype=np.float64)
    if not surf_a.any() or not surf_b.any():
        return np.array([math.inf], dtype=np.float64)
    dist_b = ndimage.distance_transform_edt(~surf_b, sampling=spacing)
    dist_a = ndimage.distance_transform_edt(~surf_a, sampling=spacing)
    return np.concatenate([dist_b[surf_a], dist_a[surf_b]]).astype(np.float64)


def hausdorff_distance(mask_a: np.ndarray, mask_b: np.ndarray, spacing: tuple[float, float] | None = None) -> float:
    distances = surface_distances(mask_a, mask_b, spacing=spacing)
    return float(np.max(distances))


def hausdorff95(mask_a: np.ndarray, mask_b: np.ndarray, spacing: tuple[float, float] | None = None) -> float:
    distances = surface_distances(mask_a, mask_b, spacing=spacing)
    return float(np.percentile(distances, 95))


def average_surface_distance(mask_a: np.ndarray, mask_b: np.ndarray, spacing: tuple[float, float] | None = None) -> float:
    distances = surface_distances(mask_a, mask_b, spacing=spacing)
    return float(np.mean(distances))


def multiclass_surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
    spacing: tuple[float, float] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in range(1, num_classes):
        pred = prediction == cls
        tgt = target == cls
        out[f"hd_class_{cls}"] = hausdorff_distance(pred, tgt, spacing)
        out[f"hd95_class_{cls}"] = hausdorff95(pred, tgt, spacing)
        out[f"asd_class_{cls}"] = average_surface_distance(pred, tgt, spacing)
    finite_hd95 = [v for k, v in out.items() if k.startswith("hd95") and np.isfinite(v)]
    finite_asd = [v for k, v in out.items() if k.startswith("asd") and np.isfinite(v)]
    out["mean_hd95"] = float(np.mean(finite_hd95)) if finite_hd95 else math.inf
    out["mean_asd"] = float(np.mean(finite_asd)) if finite_asd else math.inf
    return out

