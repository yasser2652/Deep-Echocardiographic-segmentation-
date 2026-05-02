from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_datasets, create_dummy_camus_dataset
from src.losses import build_loss
from src.metrics import MetricAccumulator
from src.model_registry import build_model_from_config
from src.utils import (
    AverageMeter,
    append_csv_row,
    copy_config,
    get_device,
    load_checkpoint,
    load_config,
    make_run_dir,
    save_checkpoint,
    save_config,
    save_json,
    seed_everything,
    update_config_from_args,
)
from src.visualize import save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAMUS-style 2D echocardiography segmentation models.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", dest="data_root", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-size", dest="image_size", type=int, default=None)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", default=None)
    parser.add_argument("--use-synthetic", dest="use_synthetic", action="store_true", default=None)
    parser.add_argument("--temporal-window", dest="temporal_window", type=int, default=None)
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--create-dummy-data", action="store_true", help="Create a tiny CAMUS-like dataset for smoke runs.")
    return parser.parse_args()


def make_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("lr", 3e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    name = str(config.get("scheduler", "cosine")).lower()
    epochs = int(config.get("epochs", 100))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    if name == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))


def make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any,
    use_amp: bool,
) -> float:
    model.train()
    meter = AverageMeter()
    for images, masks, _metadata in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        meter.update(float(loss.detach().cpu()), n=images.size(0))
    return meter.avg


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    overlay_dir: Path | None = None,
    epoch: int = 0,
) -> tuple[float, dict[str, float]]:
    model.eval()
    meter = AverageMeter()
    metrics = MetricAccumulator(num_classes=num_classes)
    saved_overlay = False
    for images, masks, metadata in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        meter.update(float(loss.detach().cpu()), n=images.size(0))
        metrics.update(logits, masks)
        if overlay_dir is not None and not saved_overlay:
            pred = logits.argmax(dim=1)[0].detach().cpu().numpy()
            img = images[0].detach().cpu().numpy()
            patient = metadata["patient_id"][0] if isinstance(metadata.get("patient_id"), list) else "sample"
            save_overlay(img, pred, overlay_dir / f"epoch{epoch:03d}_{patient}.png")
            saved_overlay = True
    return meter.avg, metrics.compute()


def main() -> None:
    args = parse_args()
    config = update_config_from_args(load_config(args.config), args)
    seed_everything(int(config.get("seed", 42)))
    output_base = Path(config.get("output_dir", "outputs"))
    if args.create_dummy_data and not config.get("data_root"):
        dummy_root = output_base / "dummy_camus"
        create_dummy_camus_dataset(dummy_root, num_patients=8, image_size=int(config.get("image_size", 128)))
        config["data_root"] = str(dummy_root)
        config["epochs"] = min(int(config.get("epochs", 2)), 2)
    run_dir = make_run_dir(output_base, args.run_name)
    save_config(config, run_dir)
    copy_config(args.config, run_dir)

    device = get_device(config.get("device", "auto"))
    datasets = build_datasets(config, require_masks=True)
    if len(datasets.get("val", [])) == 0:
        datasets["val"] = datasets["test"] if len(datasets.get("test", [])) else datasets["train"]
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(config.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=pin_memory,
    )

    model = build_model_from_config(config).to(device)
    criterion = build_loss(str(config.get("loss", "dice_ce")), num_classes=int(config.get("num_classes", 4))).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    use_amp = bool(config.get("use_mixed_precision", True)) and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)
    start_epoch = 1
    best_dice = -np.inf

    if config.get("resume"):
        ckpt = load_checkpoint(config["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("metrics", {}).get("val_mean_dice", best_dice))

    overlay_dir = run_dir / "overlays" if config.get("save_overlay_examples", True) else None
    log_csv = run_dir / "training_log.csv"
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, int(config.get("epochs", 100)) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=int(config.get("num_classes", 4)),
            overlay_dir=overlay_dir,
            epoch=epoch,
        )
        val_mean_dice = float(val_metrics["mean_dice"])
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mean_dice)
            else:
                scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_dice": val_mean_dice,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_csv_row(log_csv, row)
        history.append(row)
        save_checkpoint(run_dir / "latest.pth", model, optimizer, scheduler, epoch, config, row)
        if val_mean_dice > best_dice:
            best_dice = val_mean_dice
            save_checkpoint(run_dir / "best.pth", model, optimizer, scheduler, epoch, config, row)
        save_json({"best_val_mean_dice": best_dice, "last": row, "history": history}, run_dir / "training_summary.json")
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mean_dice={val_mean_dice:.4f}")

    print(f"Training complete. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
