from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.transforms import minmax_normalize
from src.utils import ensure_dir


DEFAULT_PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 57, 70],
        [42, 157, 143],
        [69, 123, 157],
        [244, 162, 97],
        [131, 56, 236],
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: np.ndarray, palette: np.ndarray = DEFAULT_PALETTE) -> np.ndarray:
    mask = np.asarray(mask).astype(np.int64)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls in np.unique(mask):
        color[mask == cls] = palette[int(cls) % len(palette)]
    return color


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[0] <= 5:
            image = image[image.shape[0] // 2]
        else:
            image = image.mean(axis=-1)
    gray = (minmax_normalize(image) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    color = colorize_mask(mask)
    foreground = mask > 0
    out = rgb.copy()
    out[foreground] = ((1 - alpha) * out[foreground] + alpha * color[foreground]).astype(np.uint8)
    return out


def save_overlay(image: np.ndarray, mask: np.ndarray, path: str | Path, alpha: float = 0.45) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(overlay_mask(image, mask, alpha=alpha)).save(path)


def save_color_mask(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(colorize_mask(mask)).save(path)


def plot_training_curves(csv_path: str | Path, output_path: str | Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if "train_loss" in df:
        axes[0].plot(df["epoch"], df["train_loss"], label="train")
    if "val_loss" in df:
        axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    if "val_mean_dice" in df:
        axes[1].plot(df["epoch"], df["val_mean_dice"], label="mean Dice")
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
