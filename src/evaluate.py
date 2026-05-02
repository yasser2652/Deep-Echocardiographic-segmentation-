from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.clinical_metrics import clinical_metric_errors, estimate_patient_volumes, save_bland_altman_plot
from src.dataset import build_datasets
from src.metrics import batch_segmentation_metrics
from src.model_registry import build_model_from_config
from src.postprocess import postprocess_mask
from src.surface_metrics import multiclass_surface_metrics
from src.utils import ensure_dir, get_device, load_checkpoint, load_config, save_json, update_config_from_args
from src.visualize import save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation checkpoint.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", dest="data_root", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=None)
    return parser.parse_args()


def _spacing_from_metadata(metadata: dict[str, Any], idx: int = 0) -> tuple[float, float]:
    spacing = metadata.get("spacing", (1.0, 1.0))
    if isinstance(spacing, list):
        if len(spacing) == 2 and torch.is_tensor(spacing[0]):
            return float(spacing[0][idx]), float(spacing[1][idx])
        if len(spacing) > idx and isinstance(spacing[idx], (list, tuple)):
            return float(spacing[idx][0]), float(spacing[idx][1])
    if torch.is_tensor(spacing):
        vals = spacing[idx].tolist()
        return float(vals[0]), float(vals[1])
    return (1.0, 1.0)


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    config = update_config_from_args(dict(ckpt.get("config") or load_config(args.config)), args)
    output_dir = ensure_dir(args.output_dir or Path(args.checkpoint).with_suffix("").parent / "evaluation")
    device = get_device(config.get("device", "auto"))
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    datasets = build_datasets(config, require_masks=True)
    dataset = datasets.get(args.split) or datasets.get("val") or datasets["train"]
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    num_classes = int(config.get("num_classes", 4))
    pp_cfg = config.get("postprocessing", {}) if config.get("use_postprocessing", True) else None
    rows = []
    pred_by_patient: dict[str, dict[tuple[str, str], np.ndarray]] = defaultdict(dict)
    ref_by_patient: dict[str, dict[tuple[str, str], np.ndarray]] = defaultdict(dict)
    spacing_by_patient: dict[str, dict[tuple[str, str], tuple[float, float]]] = defaultdict(dict)
    for batch_idx, (images, masks, metadata) in enumerate(tqdm(loader, desc="evaluate")):
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        targets = masks.numpy()
        for i in range(preds.shape[0]):
            pred = preds[i]
            if pp_cfg is not None:
                pred = postprocess_mask(pred, num_classes=num_classes, config=pp_cfg)
            target = targets[i]
            patient = metadata["patient_id"][i]
            view = metadata["view"][i]
            phase = metadata["phase"][i]
            spacing = _spacing_from_metadata(metadata, i)
            seg = batch_segmentation_metrics(pred[None], target[None], num_classes=num_classes)
            surf = multiclass_surface_metrics(pred, target, num_classes=num_classes, spacing=spacing)
            row = {
                "patient_id": patient,
                "view": view,
                "phase": phase,
                "image_path": metadata["image_path"][i],
                **seg,
                **surf,
            }
            rows.append(row)
            pred_by_patient[patient][(view, phase)] = pred
            ref_by_patient[patient][(view, phase)] = target
            spacing_by_patient[patient][(view, phase)] = spacing
            if batch_idx < 8:
                save_overlay(images[i].detach().cpu().numpy(), pred, output_dir / "overlays" / f"{patient}_{view}_{phase}.png")
            if seg["mean_dice"] < 0.5:
                save_overlay(images[i].detach().cpu().numpy(), pred, output_dir / "failure_cases" / f"{patient}_{view}_{phase}.png")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    summary = metrics_df.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
    summary["num_samples"] = len(metrics_df)

    pred_clinical = []
    ref_clinical = []
    clinical_cfg = config.get("clinical", {}) or {}
    for patient in sorted(pred_by_patient):
        pred_vols = estimate_patient_volumes(
            pred_by_patient[patient],
            spacing_by_patient.get(patient),
            lv_class=int(clinical_cfg.get("lv_class", 1)),
            coefficient=float(clinical_cfg.get("area_volume_coefficient", 0.85)),
        )
        ref_vols = estimate_patient_volumes(
            ref_by_patient[patient],
            spacing_by_patient.get(patient),
            lv_class=int(clinical_cfg.get("lv_class", 1)),
            coefficient=float(clinical_cfg.get("area_volume_coefficient", 0.85)),
        )
        pred_clinical.append({"patient_id": patient, **pred_vols})
        ref_clinical.append({"patient_id": patient, **{f"ref_{k}": v for k, v in ref_vols.items()}})
    pred_df = pd.DataFrame(pred_clinical)
    ref_df = pd.DataFrame(ref_clinical)
    per_patient = pred_df.merge(ref_df, on="patient_id", how="outer") if not pred_df.empty else pd.DataFrame()
    per_patient.to_csv(output_dir / "per_patient_metrics.csv", index=False)
    ref_rows = [
        {"patient_id": row["patient_id"], "LVEDV": row.get("ref_LVEDV"), "LVESV": row.get("ref_LVESV"), "LVEF": row.get("ref_LVEF")}
        for row in ref_df.to_dict("records")
    ]
    summary.update(clinical_metric_errors(pred_clinical, ref_rows))
    save_json(summary, output_dir / "metrics.json")
    if not per_patient.empty and {"LVEF", "ref_LVEF"}.issubset(per_patient.columns):
        save_bland_altman_plot(per_patient["LVEF"].to_numpy(), per_patient["ref_LVEF"].to_numpy(), output_dir / "bland_altman_lvef.png", "LVEF Bland-Altman")
    print(f"Evaluation complete. Metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
