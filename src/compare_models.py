from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_datasets
from src.metrics import MetricAccumulator
from src.model_registry import build_model_from_config
from src.models.common import count_trainable_parameters
from src.utils import ensure_dir, get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained segmentation models on a held-out split.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names, e.g. unet gdkvm echovim osa.")
    parser.add_argument("--checkpoints", required=True, help="Checkpoint file or directory containing model checkpoints.")
    parser.add_argument("--data-dir", "--data_root", "--data-root", dest="data_root", required=True)
    parser.add_argument("--output", default="outputs/model_comparison.csv")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--image-size", dest="image_size", type=int, default=None)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=None)
    return parser.parse_args()


def find_checkpoint(checkpoints: str | Path, model_name: str) -> Path | None:
    root = Path(checkpoints)
    if root.is_file():
        return root
    candidates = [
        root / model_name / "best_dice.pt",
        root / model_name / "best.pth",
        root / model_name / "last.pt",
        root / model_name / "latest.pth",
        root / f"{model_name}_best_dice.pt",
        root / f"{model_name}_best.pth",
        root / f"{model_name}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    priority = {"best_dice.pt": 0, "best.pth": 1, "last.pt": 2, "latest.pth": 3}
    matches = []
    for candidate in root.rglob("*"):
        name = candidate.name.lower()
        if name in priority and model_name.lower() in " ".join(part.lower() for part in candidate.parts):
            matches.append(candidate)
    if not matches:
        return None
    return sorted(matches, key=lambda p: (priority[p.name.lower()], str(p)))[0]


def _config_for_eval(ckpt: dict[str, Any], model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    config = dict(ckpt.get("config") or {})
    config["model"] = model_name
    config["data_root"] = args.data_root
    config["batch_size"] = args.batch_size
    if args.image_size is not None:
        config["image_size"] = args.image_size
    if args.num_classes is not None:
        config["num_classes"] = args.num_classes
    config.setdefault("num_classes", 4)
    config.setdefault("in_channels", 1)
    config.setdefault("model_params", {"base_channels": 32, "batch_norm": True, "dropout": 0.0})
    return config


@torch.no_grad()
def evaluate_model(model_name: str, checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    config = _config_for_eval(ckpt, model_name, args)
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    datasets = build_datasets(config, require_masks=True)
    dataset = datasets.get(args.split) or datasets.get("val") or datasets["train"]
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    num_classes = int(config.get("num_classes", 4))
    metrics = MetricAccumulator(num_classes=num_classes)
    samples = 0
    elapsed = 0.0
    for images, masks, _metadata in tqdm(loader, desc=f"compare:{model_name}", leave=False):
        images = images.to(device)
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        samples += images.shape[0]
        metrics.update(logits.detach().cpu(), masks)
    summary = metrics.compute()
    return {
        "Model": model_name,
        "Dice": summary["mean_dice"],
        "IoU": summary["mean_iou"],
        "Params": count_trainable_parameters(model),
        "FPS": samples / max(elapsed, 1e-8),
        "Checkpoint": str(checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        checkpoint = find_checkpoint(args.checkpoints, model_name)
        if checkpoint is None:
            rows.append(
                {
                    "Model": model_name,
                    "Dice": np.nan,
                    "IoU": np.nan,
                    "Params": np.nan,
                    "FPS": np.nan,
                    "Checkpoint": "not found",
                }
            )
            continue
        rows.append(evaluate_model(model_name, checkpoint, args, device))
    df = pd.DataFrame(rows)
    output = Path(args.output)
    ensure_dir(output.parent)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))
    print(f"Comparison saved to {output}")


if __name__ == "__main__":
    main()
