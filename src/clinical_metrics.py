from __future__ import annotations

import warnings
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import stats


def mask_area_mm2(mask: np.ndarray, class_id: int = 1, spacing: tuple[float, float] | None = None) -> float:
    if spacing is None:
        warnings.warn("Pixel spacing is missing; assuming 1.0 mm x 1.0 mm.", RuntimeWarning, stacklevel=2)
        spacing = (1.0, 1.0)
    pixel_area = float(spacing[0]) * float(spacing[1])
    return float(np.sum(mask == class_id) * pixel_area)


def pixel_spacing_from_image_header(path: str | Path | None) -> tuple[float, float] | None:
    """Read in-plane pixel spacing as (row_mm, col_mm) when metadata is available.

    CAMUS commonly ships MHD files, but many users convert to NIfTI. This helper
    supports SimpleITK-readable files first, then falls back to nibabel for
    `.nii` and `.nii.gz`. It returns None instead of raising so clinical
    reporting can continue with explicit warnings.
    """

    if path is None:
        return None
    image_path = Path(path)
    if not image_path.exists():
        return None
    name = image_path.name.lower()
    if not name.endswith((".nii", ".nii.gz", ".mhd", ".mha")):
        return None
    try:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(image_path))
        spacing = image.GetSpacing()
        if len(spacing) >= 2:
            return float(spacing[1]), float(spacing[0])
    except Exception:
        pass
    if name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib

            zooms = nib.load(str(image_path)).header.get_zooms()
            if len(zooms) >= 2:
                return float(zooms[1]), float(zooms[0])
        except Exception:
            return None
    return None


def resolve_spacing(
    spacing: tuple[float, float] | None = None,
    image_path: str | Path | None = None,
    warnings_out: list[str] | None = None,
) -> tuple[float, float]:
    if spacing is not None:
        return float(spacing[0]), float(spacing[1])
    header_spacing = pixel_spacing_from_image_header(image_path)
    if header_spacing is not None:
        return header_spacing
    if warnings_out is not None:
        warnings_out.append(f"Pixel spacing missing for {image_path or 'mask'}; assuming 1.0 mm x 1.0 mm.")
    return 1.0, 1.0


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


def _lv_width_profile(
    mask: np.ndarray,
    class_id: int,
    spacing: tuple[float, float],
    num_disks: int = 20,
) -> tuple[np.ndarray, float]:
    coords = np.argwhere(np.asarray(mask) == class_id)
    if coords.shape[0] < 2:
        return np.zeros(num_disks, dtype=np.float64), 0.0
    # Physical coordinates are [row_mm, col_mm]. PCA gives a stable long-axis
    # direction without assuming the LV is perfectly vertical in the image.
    pts = np.column_stack([coords[:, 0] * float(spacing[0]), coords[:, 1] * float(spacing[1])]).astype(np.float64)
    centered = pts - pts.mean(axis=0, keepdims=True)
    c00 = float(np.mean(centered[:, 0] * centered[:, 0]))
    c11 = float(np.mean(centered[:, 1] * centered[:, 1]))
    c01 = float(np.mean(centered[:, 0] * centered[:, 1]))
    angle = 0.5 * math.atan2(2.0 * c01, c00 - c11)
    axis = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    long_pos = centered[:, 0] * axis[0] + centered[:, 1] * axis[1]
    short_pos = centered[:, 0] * perp[0] + centered[:, 1] * perp[1]
    lo, hi = float(long_pos.min()), float(long_pos.max())
    length = hi - lo
    if length <= 1e-6:
        return np.zeros(num_disks, dtype=np.float64), 0.0
    edges = np.linspace(lo, hi, num_disks + 1)
    widths = np.zeros(num_disks, dtype=np.float64)
    for idx in range(num_disks):
        in_slice = (long_pos >= edges[idx]) & (long_pos <= edges[idx + 1] if idx == num_disks - 1 else long_pos < edges[idx + 1])
        if np.any(in_slice):
            vals = short_pos[in_slice]
            widths[idx] = float(vals.max() - vals.min())
    if np.any(widths == 0) and np.any(widths > 0):
        good = np.where(widths > 0)[0]
        bad = np.where(widths == 0)[0]
        widths[bad] = np.interp(bad, good, widths[good])
    return widths, float(length)


def estimate_biplane_simpson_volume_ml(
    mask_2ch: np.ndarray,
    mask_4ch: np.ndarray,
    class_id: int = 1,
    spacing_2ch: tuple[float, float] | None = None,
    spacing_4ch: tuple[float, float] | None = None,
    num_disks: int = 20,
) -> float:
    """Approximate CAMUS-style biplane Simpson LV volume in milliliters.

    The LV cavity is split into disks along the long axis in both 2CH and 4CH
    masks. Each disk uses an elliptical cross-section with diameters from the
    two views, then volumes are summed. This is a research estimate, not a
    clinically validated measurement.
    """

    spacing_2ch = spacing_2ch or (1.0, 1.0)
    spacing_4ch = spacing_4ch or (1.0, 1.0)
    widths_2ch, length_2ch = _lv_width_profile(mask_2ch, class_id, spacing_2ch, num_disks=num_disks)
    widths_4ch, length_4ch = _lv_width_profile(mask_4ch, class_id, spacing_4ch, num_disks=num_disks)
    if length_2ch <= 0 or length_4ch <= 0 or not np.any(widths_2ch > 0) or not np.any(widths_4ch > 0):
        return 0.0
    long_axis = min(length_2ch, length_4ch)
    disk_height = long_axis / max(1, num_disks)
    disk_volumes_mm3 = (np.pi / 4.0) * widths_2ch * widths_4ch * disk_height
    return float(np.sum(disk_volumes_mm3) / 1000.0)


def lvef_percent(lvedv_ml: float, lvesv_ml: float) -> float:
    if lvedv_ml <= 0:
        return float("nan")
    return float((lvedv_ml - lvesv_ml) / lvedv_ml * 100.0)


def estimate_patient_volumes(
    masks_by_view_phase: dict[tuple[str, str], np.ndarray],
    spacing_by_view_phase: dict[tuple[str, str], tuple[float, float] | None] | None = None,
    image_paths_by_view_phase: dict[tuple[str, str], str | Path | None] | None = None,
    lv_class: int = 1,
    coefficient: float = 0.85,
    method: str = "simpson",
    num_disks: int = 20,
    plausible_lvef_range: tuple[float, float] = (0.0, 90.0),
) -> dict[str, float]:
    spacing_by_view_phase = spacing_by_view_phase or defaultdict(lambda: None)
    image_paths_by_view_phase = image_paths_by_view_phase or defaultdict(lambda: None)
    out: dict[str, Any] = {}
    warnings_out: list[str] = []
    required = [("2CH", "ED"), ("4CH", "ED"), ("2CH", "ES"), ("4CH", "ES")]
    for key in required:
        if key not in masks_by_view_phase:
            warnings_out.append(f"Missing required {key[0]} {key[1]} mask for biplane clinical metrics.")
    for phase, key in (("ED", "LVEDV"), ("ES", "LVESV")):
        m2 = masks_by_view_phase.get(("2CH", phase))
        m4 = masks_by_view_phase.get(("4CH", phase))
        if m2 is not None and m4 is not None:
            if not np.any(np.asarray(m2) == lv_class):
                warnings_out.append(f"Empty LV cavity in 2CH {phase} mask.")
            if not np.any(np.asarray(m4) == lv_class):
                warnings_out.append(f"Empty LV cavity in 4CH {phase} mask.")
            spacing_2ch = resolve_spacing(spacing_by_view_phase.get(("2CH", phase)), image_paths_by_view_phase.get(("2CH", phase)), warnings_out)
            spacing_4ch = resolve_spacing(spacing_by_view_phase.get(("4CH", phase)), image_paths_by_view_phase.get(("4CH", phase)), warnings_out)
            if str(method).lower() in {"simpson", "biplane_simpson", "camus_simpson"}:
                out[key] = estimate_biplane_simpson_volume_ml(
                    m2,
                    m4,
                    class_id=lv_class,
                    spacing_2ch=spacing_2ch,
                    spacing_4ch=spacing_4ch,
                    num_disks=num_disks,
                )
            else:
                out[key] = estimate_biplane_area_length_volume_ml(
                    m2,
                    m4,
                    class_id=lv_class,
                    spacing_2ch=spacing_2ch,
                    spacing_4ch=spacing_4ch,
                )
        else:
            available = m2 if m2 is not None else m4
            view = "2CH" if m2 is not None else "4CH"
            spacing = spacing_by_view_phase.get((view, phase))
            image_path = image_paths_by_view_phase.get((view, phase))
            if available is None:
                out[key] = float("nan")
                warnings_out.append(f"No LV mask available for {phase}; cannot estimate {key}.")
            else:
                msg = f"Only one {phase} view is available; using approximate area-based volume."
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                warnings_out.append(msg)
                out[key] = estimate_area_based_volume_ml(
                    available,
                    class_id=lv_class,
                    spacing=resolve_spacing(spacing, image_path, warnings_out),
                    coefficient=coefficient,
                )
    out["LVEF"] = lvef_percent(out.get("LVEDV", float("nan")), out.get("LVESV", float("nan")))
    if np.isfinite(out.get("LVEDV", np.nan)) and np.isfinite(out.get("LVESV", np.nan)) and out["LVEDV"] <= out["LVESV"]:
        warnings_out.append("Sanity check failed: LVEDV is not greater than LVESV.")
    if np.isfinite(out.get("LVEF", np.nan)):
        lo, hi = plausible_lvef_range
        if out["LVEF"] < lo or out["LVEF"] > hi:
            warnings_out.append(f"Sanity check failed: LVEF {out['LVEF']:.2f}% is outside plausible range [{lo}, {hi}].")
    out["clinical_warnings"] = warnings_out
    out["clinical_warnings_text"] = "; ".join(warnings_out)
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


def save_patient_clinical_summary(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_rows: list[dict[str, Any]] = []

    def clean_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [clean_value(v) for v in value]
        return value

    for row in rows:
        clean = {key: clean_value(value) for key, value in row.items()}
        if isinstance(clean.get("clinical_warnings"), list):
            clean["clinical_warnings"] = "; ".join(str(v) for v in clean["clinical_warnings"])
        if isinstance(clean.get("ref_clinical_warnings"), list):
            clean["ref_clinical_warnings"] = "; ".join(str(v) for v in clean["ref_clinical_warnings"])
        serializable_rows.append(clean)
    if serializable_rows:
        keys = sorted({key for row in serializable_rows for key in row})
        with (output_dir / "patient_clinical_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(serializable_rows)
    else:
        (output_dir / "patient_clinical_metrics.csv").write_text("", encoding="utf-8")
    with (output_dir / "patient_clinical_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_rows, f, indent=2)


def save_bland_altman_plot(predicted: np.ndarray, reference: np.ndarray, path: str, title: str = "Bland-Altman") -> None:
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    valid = np.isfinite(predicted) & np.isfinite(reference)
    mean = (predicted[valid] + reference[valid]) / 2.0
    diff = predicted[valid] - reference[valid]
    bias = np.mean(diff) if diff.size else np.nan
    sd = np.std(diff, ddof=1) if diff.size > 1 else 0.0
    width, height = 960, 640
    margin_left, margin_top, margin_right, margin_bottom = 90, 70, 40, 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot = (margin_left, margin_top, width - margin_right, height - margin_bottom)
    draw.rectangle(plot, outline=(30, 30, 30), width=2)
    draw.text((margin_left, 25), title, fill=(20, 20, 20))
    draw.text((width // 2 - 30, height - 38), "Mean", fill=(20, 20, 20))
    draw.text((15, height // 2), "Pred - ref", fill=(20, 20, 20))

    if mean.size:
        x_min, x_max = float(np.min(mean)), float(np.max(mean))
        y_values = np.concatenate([diff, np.array([bias, bias + 1.96 * sd, bias - 1.96 * sd])])
        y_values = y_values[np.isfinite(y_values)]
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        def project(x_val: float, y_val: float) -> tuple[int, int]:
            x0, y0, x1, y1 = plot
            px = x0 + int((x_val - x_min) / max(1e-8, x_max - x_min) * (x1 - x0))
            py = y1 - int((y_val - y_min) / max(1e-8, y_max - y_min) * (y1 - y0))
            return px, py

        for x_val, y_val in zip(mean, diff):
            px, py = project(float(x_val), float(y_val))
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(40, 115, 190))
        for value, color in (
            (bias, (20, 20, 20)),
            (bias + 1.96 * sd, (190, 45, 45)),
            (bias - 1.96 * sd, (190, 45, 45)),
        ):
            if np.isfinite(value):
                _, py = project(x_min, float(value))
                draw.line((plot[0], py, plot[2], py), fill=color, width=2)
        draw.text((plot[0], plot[3] + 8), f"{x_min:.2f}", fill=(20, 20, 20))
        draw.text((plot[2] - 50, plot[3] + 8), f"{x_max:.2f}", fill=(20, 20, 20))
        draw.text((plot[0] - 70, plot[1]), f"{y_max:.2f}", fill=(20, 20, 20))
        draw.text((plot[0] - 70, plot[3] - 12), f"{y_min:.2f}", fill=(20, 20, 20))
        draw.text((plot[2] - 230, plot[1] + 12), f"bias={bias:.2f}, sd={sd:.2f}", fill=(20, 20, 20))
    else:
        draw.text((margin_left + 20, margin_top + 30), "No finite paired values available.", fill=(120, 30, 30))

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
