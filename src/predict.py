from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from src.dataset import infer_phase, infer_view, load_medical_image, select_2d
from src.model_registry import build_model_from_config
from src.transforms import SegmentationTransform
from src.utils import ensure_dir, get_device, load_checkpoint
from src.visualize import save_color_mask, save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict segmentation masks for one image, a folder, or a patient folder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-confidence", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    return parser.parse_args()


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    supported = []
    for file in path.rglob("*"):
        if not file.is_file():
            continue
        name = file.name.lower()
        if any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz", ".mhd", ".nii", ".nii.gz")):
            if not any(token in name for token in ("_gt", "mask", "seg", "label")):
                supported.append(file)
    return sorted(supported)


def quality_checks(mask: np.ndarray, probs: np.ndarray | None = None) -> list[str]:
    flags: list[str] = []
    lv = mask == 1
    myo = mask == 2
    if np.sum(mask > 0) == 0:
        flags.append("empty_mask")
    labels, count = ndimage.label(lv)
    if count > 1:
        sizes = np.bincount(labels.ravel())[1:]
        if len(sizes) > 1 and np.sort(sizes)[-2] > 25:
            flags.append("disconnected_lv_cavity")
    lv_fraction = float(lv.mean())
    if lv_fraction < 0.005 or lv_fraction > 0.60:
        flags.append("abnormal_lv_area")
    if lv.any():
        ratio = float(myo.sum() / max(1, lv.sum()))
        if ratio < 0.15 or ratio > 4.0:
            flags.append("abnormal_myocardium_lv_ratio")
    if lv.any() and myo.any():
        dist_to_myo = ndimage.distance_transform_edt(~myo)
        if float(np.percentile(dist_to_myo[lv], 75)) > 10:
            flags.append("impossible_anatomy_layout")
    if probs is not None:
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-7, 1.0)), axis=0) / np.log(probs.shape[0])
        if float(np.mean(entropy)) > 0.65:
            flags.append("low_confidence_prediction")
    return flags


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    device = get_device(args.device)
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    output_dir = ensure_dir(args.output_dir)
    image_size = int(config.get("image_size", 256))
    prep = config.get("preprocessing", {}) or {}
    transform = SegmentationTransform(image_size=image_size, training=False, normalize=prep.get("normalize", "minmax"), z_score=bool(prep.get("z_score", False)))
    temporal_window = int(config.get("temporal_window", 1))
    num_classes = int(config.get("num_classes", 4))
    input_paths = collect_inputs(Path(args.input))
    if not input_paths:
        raise FileNotFoundError(f"No supported image files found at {args.input}")
    all_flags = []
    phase_areas: dict[tuple[str, str], float] = {}
    for image_path in input_paths:
        image = select_2d(load_medical_image(image_path))
        image_tensor, _ = transform(image, np.zeros_like(image, dtype=np.int64))
        if temporal_window > image_tensor.shape[0]:
            image_tensor = image_tensor.repeat(temporal_window, 1, 1)
        x = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = probs.argmax(axis=0).astype(np.uint8)
        if args.postprocess:
            from src.postprocess import postprocess_mask

            pred = postprocess_mask(pred, num_classes=num_classes, config=config.get("postprocessing", {})).astype(np.uint8)
        stem = image_path.stem.replace(".nii", "")
        np.save(output_dir / f"{stem}_mask.npy", pred)
        Image.fromarray(pred).save(output_dir / f"{stem}_mask.png")
        save_color_mask(pred, output_dir / f"{stem}_color_mask.png")
        save_overlay(image_tensor.numpy(), pred, output_dir / f"{stem}_overlay.png")
        if args.save_confidence:
            np.save(output_dir / f"{stem}_softmax.npy", probs)
        flags = quality_checks(pred, probs)
        view = infer_view(image_path) or "unknown"
        phase = infer_phase(image_path) or "unknown"
        if phase in {"ED", "ES"}:
            phase_areas[(view, phase)] = float(np.sum(pred == 1))
        if flags:
            failure_dir = ensure_dir(output_dir / "failures")
            save_overlay(image_tensor.numpy(), pred, failure_dir / f"{stem}_overlay.png")
        all_flags.append({"image_path": str(image_path), "flags": flags})
    for view in ("2CH", "4CH", "unknown"):
        ed = phase_areas.get((view, "ED"))
        es = phase_areas.get((view, "ES"))
        if ed is not None and es is not None and ed < es:
            all_flags.append({"image_path": view, "flags": ["ed_area_smaller_than_es_area"]})
    with (output_dir / "quality_flags.json").open("w", encoding="utf-8") as f:
        json.dump(all_flags, f, indent=2)
    print(f"Predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
