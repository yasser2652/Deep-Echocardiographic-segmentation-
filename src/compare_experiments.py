from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.utils import ensure_dir


CLASS_LABELS = {
    1: "LV cavity",
    2: "Myocardium",
    3: "Left atrium",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple experiment output folders.")
    parser.add_argument("experiment_dirs", nargs="+")
    parser.add_argument("--output-dir", default="outputs/comparison")
    return parser.parse_args()


def read_summary(exp_dir: Path) -> dict:
    metrics_json = exp_dir / "metrics.json"
    training_summary = exp_dir / "training_summary.json"
    row = {"experiment": exp_dir.name, "path": str(exp_dir)}
    if metrics_json.exists():
        with metrics_json.open("r", encoding="utf-8") as f:
            row.update(json.load(f))
    elif training_summary.exists():
        with training_summary.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        row["best_val_mean_dice"] = summary.get("best_val_mean_dice")
        row.update({f"last_{k}": v for k, v in summary.get("last", {}).items() if isinstance(v, (int, float))})
    metrics_csv = exp_dir / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        for col in df.select_dtypes(include="number").columns:
            row[f"mean_{col}"] = df[col].mean()
    return row


def plot_curves(exp_dirs: list[Path], output_dir: Path) -> None:
    loss_series = {}
    dice_series = {}
    for exp in exp_dirs:
        csv_path = exp / "training_log.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        label = exp.name
        if "val_loss" in df:
            loss_series[label] = list(zip(df["epoch"].astype(float), df["val_loss"].astype(float)))
        if "val_mean_dice" in df:
            dice_series[label] = list(zip(df["epoch"].astype(float), df["val_mean_dice"].astype(float)))
    if loss_series:
        draw_line_plot(loss_series, output_dir / "validation_loss_curves.png", "Validation Loss")
    if dice_series:
        draw_line_plot(dice_series, output_dir / "validation_dice_curves.png", "Validation Mean Dice", y_range=(0.0, 1.0))


def _metric_value(row: pd.Series, candidates: list[str]) -> float:
    for key in candidates:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return float("nan")


def plot_dice_per_structure(summary_df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for _, row in summary_df.iterrows():
        for class_id, label in CLASS_LABELS.items():
            value = _metric_value(row, [f"dice_class_{class_id}", f"mean_dice_class_{class_id}", f"mean_dice_class_{class_id}.0"])
            if np.isfinite(value):
                rows.append({"experiment": row["experiment"], "structure": label, "dice": value})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot(index="experiment", columns="structure", values="dice")
    draw_grouped_bar_plot(pivot, output_dir / "dice_per_structure.png", "Dice per Structure", y_range=(0.0, 1.0))


def draw_line_plot(
    series: dict[str, list[tuple[float, float]]],
    path: Path,
    title: str,
    y_range: tuple[float, float] | None = None,
) -> None:
    width, height = 900, 520
    margin_l, margin_r, margin_t, margin_b = 70, 180, 50, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 18), title, fill="black")
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    all_points = [point for points in series.values() for point in points if np.isfinite(point[0]) and np.isfinite(point[1])]
    if not all_points:
        image.save(path)
        return
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_max = x_min + 1
    y_min, y_max = y_range if y_range else (min(ys), max(ys))
    if y_min == y_max:
        y_max = y_min + 1
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline="black")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for idx, (label, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        coords = []
        for x, y in points:
            px = margin_l + (x - x_min) / (x_max - x_min) * plot_w
            py = margin_t + plot_h - (y - y_min) / (y_max - y_min) * plot_h
            coords.append((px, py))
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=3)
        for px, py in coords:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        legend_y = margin_t + idx * 22
        draw.rectangle((width - margin_r + 20, legend_y + 4, width - margin_r + 34, legend_y + 18), fill=color)
        draw.text((width - margin_r + 42, legend_y), label, fill="black")
    draw.text((margin_l + plot_w // 2 - 20, height - 38), "Epoch", fill="black")
    image.save(path)


def draw_grouped_bar_plot(pivot: pd.DataFrame, path: Path, title: str, y_range: tuple[float, float] = (0.0, 1.0)) -> None:
    width, height = 920, 520
    margin_l, margin_r, margin_t, margin_b = 70, 190, 50, 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 18), title, fill="black")
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_min, y_max = y_range
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline="black")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    experiments = list(pivot.index)
    structures = list(pivot.columns)
    if not experiments or not structures:
        image.save(path)
        return
    group_w = plot_w / len(experiments)
    bar_w = max(4, group_w / (len(structures) + 1))
    for exp_idx, experiment in enumerate(experiments):
        group_x = margin_l + exp_idx * group_w
        for struct_idx, structure in enumerate(structures):
            value = float(pivot.loc[experiment, structure])
            value = float(np.clip(value, y_min, y_max))
            x0 = group_x + (struct_idx + 0.5) * bar_w
            x1 = x0 + bar_w * 0.8
            y1 = margin_t + plot_h
            y0 = margin_t + plot_h - (value - y_min) / (y_max - y_min) * plot_h
            draw.rectangle((x0, y0, x1, y1), fill=colors[struct_idx % len(colors)])
        draw.text((group_x + 4, height - margin_b + 20), str(experiment)[:18], fill="black")
    for idx, structure in enumerate(structures):
        legend_y = margin_t + idx * 24
        draw.rectangle((width - margin_r + 20, legend_y + 4, width - margin_r + 36, legend_y + 20), fill=colors[idx % len(colors)])
        draw.text((width - margin_r + 44, legend_y), str(structure), fill="black")
    draw.text((18, margin_t + plot_h // 2), "Dice", fill="black")
    image.save(path)


def best_row(df: pd.DataFrame, candidates: list[str], maximize: bool = True) -> dict | None:
    for column in candidates:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().any():
                idx = series.idxmax() if maximize else series.idxmin()
                return df.loc[idx].to_dict()
    return None


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    exp_dirs = [Path(p) for p in args.experiment_dirs]
    rows = [read_summary(exp) for exp in exp_dirs]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "comparison_report.csv", index=False)
    report = {}
    best_dice = best_row(df, ["mean_dice", "mean_mean_dice", "best_val_mean_dice"], maximize=True)
    best_lvef = best_row(df, ["LVEF_mae", "mean_LVEF_mae"], maximize=False)
    if best_dice is not None:
        report["best_by_mean_dice"] = best_dice
    if best_lvef is not None:
        report["best_by_lvef_mae"] = best_lvef
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    plot_curves(exp_dirs, output_dir)
    plot_dice_per_structure(df, output_dir)
    print(f"Comparison saved to {output_dir / 'comparison_report.csv'}")


if __name__ == "__main__":
    main()
