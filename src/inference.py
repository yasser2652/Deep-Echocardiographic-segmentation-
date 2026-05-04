from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.dataset import load_medical_image, select_2d
from src.model_registry import build_model_from_config
from src.transforms import SegmentationTransform
from src.utils import ensure_dir, get_device, load_checkpoint
from src.visualize import save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image segmentation inference.")
    parser.add_argument("--model", default=None, help="Model name override, e.g. unet, gdkvm, echovim, osa.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", default=None, help="Optional overlay PNG path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", dest="image_size", type=int, default=None)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=None)
    parser.add_argument("--in-channels", dest="in_channels", type=int, default=None)
    return parser.parse_args()


def _config_from_checkpoint(ckpt: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(ckpt.get("config") or {})
    config.setdefault("model", args.model or "baseline_unet")
    config.setdefault("image_size", args.image_size or 256)
    config.setdefault("num_classes", args.num_classes or 4)
    config.setdefault("in_channels", args.in_channels or 1)
    if args.model is not None:
        config["model"] = args.model
    if args.image_size is not None:
        config["image_size"] = args.image_size
    if args.num_classes is not None:
        config["num_classes"] = args.num_classes
    if args.in_channels is not None:
        config["in_channels"] = args.in_channels
    config.setdefault("model_params", {"base_channels": 32, "batch_norm": True, "dropout": 0.0})
    return config


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    config = _config_from_checkpoint(ckpt, args)
    device = get_device(args.device)
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    image_np = select_2d(load_medical_image(args.image))
    prep = config.get("preprocessing", {}) or {}
    transform = SegmentationTransform(
        image_size=int(config.get("image_size", 256)),
        training=False,
        normalize=prep.get("normalize", "minmax"),
        z_score=bool(prep.get("z_score", False)),
    )
    image_tensor, _ = transform(image_np, np.zeros_like(image_np, dtype=np.int64))
    required_channels = int(config.get("in_channels", image_tensor.shape[0]))
    if image_tensor.shape[0] < required_channels:
        image_tensor = image_tensor.repeat(required_channels, 1, 1)[:required_channels]
    x = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    Image.fromarray(pred).save(output_path)
    if args.overlay:
        save_overlay(image_tensor.numpy(), pred, args.overlay)
    print(f"Saved predicted mask to {output_path}")


if __name__ == "__main__":
    main()
