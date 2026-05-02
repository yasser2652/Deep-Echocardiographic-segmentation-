import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.compare_experiments import plot_dice_per_structure, read_summary
from src.cross_validate import aggregate_fold_metrics, write_fold_configs
from src.dataset import build_datasets, create_dummy_camus_dataset
from src.predict import quality_checks
from src.surface_metrics import average_surface_distance, hausdorff95, hausdorff_distance
from src.utils import ensure_dir


def artifact_dir(name: str) -> Path:
    return ensure_dir(Path("outputs") / "test_artifacts" / f"{name}_{uuid4().hex}")


def test_temporal_dataset_uses_sequence_when_available():
    root = create_dummy_camus_dataset(artifact_dir("temporal") / "data", num_patients=4, image_size=32)
    config = {
        "data_root": str(root),
        "image_size": 32,
        "num_classes": 4,
        "num_workers": 0,
        "seed": 4,
        "num_folds": 4,
        "fold": 0,
        "temporal_window": 3,
        "augmentation": {"enabled": False},
        "preprocessing": {"normalize": "minmax", "z_score": False, "class_mapping": None},
    }
    datasets = build_datasets(config, require_masks=True)
    image, mask, metadata = datasets["train"][0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert metadata["sequence_path"] is not None
    assert not np.allclose(image[0].numpy(), image[-1].numpy())


def test_cross_validation_config_generation_and_aggregation():
    root = artifact_dir("cv")
    config = {"data_root": "CAMUS", "output_dir": str(root), "num_folds": 2, "model": "baseline_unet"}
    config_paths = write_fold_configs(config, root)
    assert len(config_paths) == 2
    for idx in range(2):
        fold_dir = ensure_dir(root / f"fold_{idx}" / "evaluation")
        with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"mean_dice": 0.7 + idx * 0.1, "LVEF_mae": 5.0 - idx}, f)
    per_fold, summary = aggregate_fold_metrics([root / "fold_0" / "evaluation", root / "fold_1" / "evaluation"], root)
    assert len(per_fold) == 2
    mean_dice = summary.loc[summary["metric"] == "mean_dice", "mean"].iloc[0]
    assert np.isclose(mean_dice, 0.75)
    assert (root / "cross_validation_summary.csv").exists()


def test_compare_experiments_reads_metrics_and_plots_dice_per_structure():
    root = artifact_dir("compare")
    exp = ensure_dir(root / "baseline")
    with (exp / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"mean_dice": 0.8, "dice_class_1": 0.85, "dice_class_2": 0.75, "dice_class_3": 0.8}, f)
    row = read_summary(exp)
    assert row["mean_dice"] == 0.8
    plot_dice_per_structure(pd.DataFrame([row]), root)
    assert (root / "dice_per_structure.png").exists()


def test_surface_metrics_identical_masks_are_zero():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:20, 8:20] = True
    assert hausdorff_distance(mask, mask) == 0.0
    assert hausdorff95(mask, mask) == 0.0
    assert average_surface_distance(mask, mask) == 0.0


def test_prediction_quality_checks_flag_obvious_failures():
    empty = np.zeros((32, 32), dtype=np.uint8)
    assert "empty_mask" in quality_checks(empty)

    disconnected = np.zeros((64, 64), dtype=np.uint8)
    disconnected[8:18, 8:18] = 1
    disconnected[42:54, 42:54] = 1
    disconnected[20:36, 20:36] = 2
    flags = quality_checks(disconnected)
    assert "disconnected_lv_cavity" in flags

