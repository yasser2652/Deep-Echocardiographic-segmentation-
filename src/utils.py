from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import torch
import yaml


PathLike = str | os.PathLike[str]


def ensure_dir(path: PathLike) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_config(path: PathLike | None = None) -> dict[str, Any]:
    if path is None:
        path = "config.yaml"
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_config(config: dict[str, Any], output_dir: PathLike) -> Path:
    output = ensure_dir(output_dir) / "config_resolved.yaml"
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return output


def copy_config(config_path: PathLike | None, output_dir: PathLike) -> None:
    if config_path is None:
        return
    src = Path(config_path)
    if src.exists():
        shutil.copy2(src, ensure_dir(output_dir) / src.name)


def update_config_from_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    arg_values = vars(args)
    for key, value in arg_values.items():
        if key == "config" or value is None:
            continue
        if key == "image_size":
            updated[key] = int(value)
        elif key == "mixed_precision":
            updated["use_mixed_precision"] = bool(value)
        elif key == "use_synthetic":
            updated["use_synthetic"] = bool(value)
        else:
            updated[key] = value
    if updated.get("learning_rate") is not None and arg_values.get("lr") is None:
        updated["lr"] = float(updated["learning_rate"])
    if updated.get("use_amp") is not None and arg_values.get("mixed_precision") is None:
        updated["use_mixed_precision"] = bool(updated["use_amp"])
    if updated.get("save_dir") is not None and arg_values.get("output_dir") is None:
        if updated.get("output_dir") in (None, "outputs") or str(updated.get("save_dir")) != "outputs":
            updated["output_dir"] = updated["save_dir"]
    return updated


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str | None = "auto") -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(output_dir: PathLike, name: str | None = None) -> Path:
    output = ensure_dir(output_dir)
    run_name = name or f"run-{timestamp()}"
    return ensure_dir(output / run_name)


def save_json(data: dict[str, Any], path: PathLike) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=convert)


def append_csv_row(path: PathLike, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_torch_save(obj: Any, path: PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    torch.save(obj, tmp)
    try:
        tmp.replace(path)
    except PermissionError:
        # Some Windows/Drive/antivirus combinations briefly block atomic
        # replace. Keep the safe temp write, then fall back to a regular copy.
        shutil.copy2(tmp, path)
        try:
            tmp.unlink(missing_ok=True)
        except PermissionError:
            pass


def save_checkpoint(
    path: PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    safe_torch_save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: PathLike, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=map_location)


def split_patients(
    patient_ids: Iterable[str],
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, set[str]]:
    ids = sorted(set(patient_ids))
    if not ids:
        return {"train": set(), "val": set(), "test": set()}
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(n * train_fraction))) if n >= 3 else max(1, n - 1)
    n_val = int(round(n * val_fraction))
    if n >= 3:
        n_val = max(1, n_val)
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    train = set(ids[:n_train])
    val = set(ids[n_train : n_train + n_val])
    test = set(ids[n_train + n_val :])
    if not test and len(ids) > 2:
        moved = next(iter(val or train))
        val.discard(moved)
        train.discard(moved)
        test.add(moved)
    return {"train": train, "val": val, "test": test}


def kfold_patient_split(patient_ids: Iterable[str], fold: int = 0, num_folds: int = 5, seed: int = 42) -> dict[str, set[str]]:
    ids = sorted(set(patient_ids))
    if not ids:
        return {"train": set(), "val": set(), "test": set()}
    num_folds = max(2, min(num_folds, len(ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    folds = np.array_split(np.array(ids), num_folds)
    fold = fold % num_folds
    val_idx = fold
    test_idx = (fold + 1) % num_folds
    val = set(folds[val_idx].tolist())
    test = set(folds[test_idx].tolist())
    train = set(ids) - val - test
    if not train:
        train = set(ids) - val
    return {"train": train, "val": val, "test": test}


def read_split_file(path: PathLike) -> dict[str, set[str]]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    splits = {"train": set(), "val": set(), "test": set()}
    if split_path.is_dir():
        for split in splits:
            candidates = [split_path / f"{split}_patients.txt", split_path / f"{split}.txt"]
            existing = next((candidate for candidate in candidates if candidate.exists()), None)
            if existing is None:
                continue
            with existing.open("r", encoding="utf-8") as f:
                for line in f:
                    patient = line.strip()
                    if patient and not patient.startswith("#"):
                        splits[split].add(patient)
        return splits
    if split_path.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(split_path)
        lower = {c.lower(): c for c in df.columns}
        patient_col = lower.get("patient_id") or lower.get("patient") or lower.get("id")
        split_col = lower.get("split") or lower.get("fold")
        if patient_col is None or split_col is None:
            raise ValueError("CSV split file must contain patient_id and split columns.")
        for _, row in df.iterrows():
            split = str(row[split_col]).lower()
            if split in splits:
                splits[split].add(str(row[patient_col]))
        return splits
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) == 1:
                raise ValueError(
                    f"Invalid split line: {line}. Single-patient lines are supported in "
                    "train_patients.txt/val_patients.txt/test_patients.txt inside a split directory."
                )
            split, patient = parts[0].lower(), parts[1]
            if split not in splits:
                raise ValueError(f"Unknown split '{split}' in {split_path}")
            splits[split].add(patient)
    return splits


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
