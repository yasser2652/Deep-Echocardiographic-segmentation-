from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils import ensure_dir, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fold configs and aggregate cross-validation metrics.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs/cross_validation")
    parser.add_argument("--data-root", dest="data_root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-folds", dest="num_folds", type=int, default=None)
    parser.add_argument("--generate-configs", action="store_true", help="Write one config YAML per fold.")
    parser.add_argument("--fold-dirs", nargs="*", default=None, help="Fold output folders containing metrics.json.")
    return parser.parse_args()


def write_fold_configs(config: dict[str, Any], output_dir: str | Path) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    num_folds = int(config.get("num_folds", 5))
    paths = []
    for fold in range(num_folds):
        fold_config = dict(config)
        fold_config["fold"] = fold
        fold_config["output_dir"] = str(output_dir / f"fold_{fold}")
        path = output_dir / f"fold_{fold}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(fold_config, f, sort_keys=False)
        paths.append(path)
    return paths


def discover_fold_dirs(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    candidates = [p.parent for p in root.rglob("metrics.json") if p.is_file()]
    return sorted(set(candidates), key=lambda p: str(p))


def read_fold_metrics(fold_dir: str | Path) -> dict[str, Any]:
    fold_dir = Path(fold_dir)
    metrics_path = fold_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Fold metrics not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    row = {"fold_dir": str(fold_dir), "fold": fold_dir.name}
    row.update(metrics)
    return row


def aggregate_fold_metrics(fold_dirs: list[str | Path], output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    rows = [read_fold_metrics(path) for path in fold_dirs]
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(output_dir / "fold_metrics.csv", index=False)
    numeric = per_fold.select_dtypes(include=[np.number])
    summary_rows = []
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": column,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if np.sum(np.isfinite(values)) > 1 else 0.0,
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "num_folds": int(np.sum(np.isfinite(values))),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "cross_validation_summary.csv", index=False)
    save_json({"folds": rows, "summary": summary.to_dict("records")}, output_dir / "cross_validation_summary.json")
    return per_fold, summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.model is not None:
        config["model"] = args.model
    if args.num_folds is not None:
        config["num_folds"] = args.num_folds
    output_dir = ensure_dir(args.output_dir)
    if args.generate_configs:
        paths = write_fold_configs(config, output_dir)
        print(f"Wrote {len(paths)} fold config files to {output_dir}")
    fold_dirs = [Path(p) for p in args.fold_dirs] if args.fold_dirs else discover_fold_dirs(output_dir)
    if fold_dirs:
        aggregate_fold_metrics(fold_dirs, output_dir)
        print(f"Aggregated {len(fold_dirs)} folds into {output_dir}")
    elif not args.generate_configs:
        print(f"No fold metrics found under {output_dir}")


if __name__ == "__main__":
    main()

