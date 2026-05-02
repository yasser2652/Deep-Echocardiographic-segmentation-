from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def mask_area_mm2(mask: np.ndarray, class_id: int = 1, spacing: tuple[float, float] | None = None) -> float:
    if spacing is None:
        warnings.warn("Pixel spacing is missing; assuming 1.0 mm x 1.0 mm.", RuntimeWarning, stacklevel=2)
        spacing = (1.0, 1.0)
    pixel_area = float(spacing[0]) * float(spacing[1])
    return float(np.sum(mask == class_id) * pixel_area)


def long_axis_length_mm(mask: np.ndarray, class_id: int = 1, spacing: tuple[float, float] | None = None) -> float:
    if spacing is None:
        spacing = (1.0, 1.0)
    coords = np.argwhere(mask == class_id)
    if coords.size == 0:
        return 0.0
    height = (coords[:, 0].max() - coords[:, 0].min() + 1) * float(spacing[0])
    width = (coords[:, 1].max() - coords[:, 1].min() + 1) * float(spacing[1])
    return float(max(height, width))


def estimate_area_based_volume_ml(
    mask: np.ndarray,
    class_id: int = 1,
    spacing: tuple[float, float] | None = None,
    coefficient: float = 0.85,
) -> float:
    area = mask_area_mm2(mask, class_id=class_id, spacing=spacing)
    if area <= 0:
        return 0.0
    # Approximation for single-plane masks; useful for trend analysis, not clinical measurement.
    return float(coefficient * (area ** 1.5) / 1000.0)


def estimate_biplane_area_length_volume_ml(
    mask_2ch: np.ndarray,
    mask_4ch: np.ndarray,
    class_id: int = 1,
    spacing_2ch: tuple[float, float] | None = None,
    spacing_4ch: tuple[float, float] | None = None,
) -> float:
    a2 = mask_area_mm2(mask_2ch, class_id=class_id, spacing=spacing_2ch)
    a4 = mask_area_mm2(mask_4ch, class_id=class_id, spacing=spacing_4ch)
    l2 = long_axis_length_mm(mask_2ch, class_id=class_id, spacing=spacing_2ch)
    l4 = long_axis_length_mm(mask_4ch, class_id=class_id, spacing=spacing_4ch)
    length = max(1e-6, min(l2, l4) if l2 > 0 and l4 > 0 else max(l2, l4))
    if a2 <= 0 or a4 <= 0:
        return 0.0
    return float((8.0 * a2 * a4) / (3.0 * np.pi * length) / 1000.0)


def lvef_percent(lvedv_ml: float, lvesv_ml: float) -> float:
    if lvedv_ml <= 0:
        return float("nan")
    return float((lvedv_ml - lvesv_ml) / lvedv_ml * 100.0)


def estimate_patient_volumes(
    masks_by_view_phase: dict[tuple[str, str], np.ndarray],
    spacing_by_view_phase: dict[tuple[str, str], tuple[float, float] | None] | None = None,
    lv_class: int = 1,
    coefficient: float = 0.85,
) -> dict[str, float]:
    spacing_by_view_phase = spacing_by_view_phase or defaultdict(lambda: None)
    out: dict[str, float] = {}
    for phase, key in (("ED", "LVEDV"), ("ES", "LVESV")):
        m2 = masks_by_view_phase.get(("2CH", phase))
        m4 = masks_by_view_phase.get(("4CH", phase))
        if m2 is not None and m4 is not None:
            out[key] = estimate_biplane_area_length_volume_ml(
                m2,
                m4,
                class_id=lv_class,
                spacing_2ch=spacing_by_view_phase.get(("2CH", phase)),
                spacing_4ch=spacing_by_view_phase.get(("4CH", phase)),
            )
        else:
            available = m2 if m2 is not None else m4
            spacing = spacing_by_view_phase.get(("2CH", phase)) if m2 is not None else spacing_by_view_phase.get(("4CH", phase))
            if available is None:
                out[key] = float("nan")
            else:
                warnings.warn(
                    f"Only one {phase} view is available; using approximate area-based volume.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                out[key] = estimate_area_based_volume_ml(
                    available,
                    class_id=lv_class,
                    spacing=spacing,
                    coefficient=coefficient,
                )
    out["LVEF"] = lvef_percent(out.get("LVEDV", float("nan")), out.get("LVESV", float("nan")))
    return out


def error_statistics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    valid = np.isfinite(predicted) & np.isfinite(reference)
    if not valid.any():
        return {"mae": float("nan"), "bias": float("nan"), "std": float("nan"), "pearson_r": float("nan")}
    diff = predicted[valid] - reference[valid]
    if valid.sum() > 1:
        pearson = stats.pearsonr(predicted[valid], reference[valid]).statistic
    else:
        pearson = float("nan")
    return {
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "std": float(np.std(diff, ddof=1)) if valid.sum() > 1 else 0.0,
        "pearson_r": float(pearson),
    }


def clinical_metric_errors(pred_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]]) -> dict[str, float]:
    ref_by_patient = {row["patient_id"]: row for row in ref_rows}
    pred_by_patient = {row["patient_id"]: row for row in pred_rows}
    common = sorted(set(pred_by_patient) & set(ref_by_patient))
    out: dict[str, float] = {}
    for metric in ("LVEDV", "LVESV", "LVEF"):
        stats_dict = error_statistics(
            np.array([pred_by_patient[p].get(metric, np.nan) for p in common]),
            np.array([ref_by_patient[p].get(metric, np.nan) for p in common]),
        )
        out.update({f"{metric}_{k}": v for k, v in stats_dict.items()})
    return out


def save_bland_altman_plot(predicted: np.ndarray, reference: np.ndarray, path: str, title: str = "Bland-Altman") -> None:
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    valid = np.isfinite(predicted) & np.isfinite(reference)
    mean = (predicted[valid] + reference[valid]) / 2.0
    diff = predicted[valid] - reference[valid]
    bias = np.mean(diff) if diff.size else np.nan
    sd = np.std(diff, ddof=1) if diff.size > 1 else 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean, diff, s=24, alpha=0.75)
    ax.axhline(bias, color="black", linestyle="-", label="Bias")
    ax.axhline(bias + 1.96 * sd, color="tab:red", linestyle="--", label="95% limits")
    ax.axhline(bias - 1.96 * sd, color="tab:red", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Mean")
    ax.set_ylabel("Prediction - reference")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
