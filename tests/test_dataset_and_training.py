from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import build_datasets, create_dummy_camus_dataset, discover_camus_samples, discover_image_mask_samples
from src.losses import build_loss
from src.model_registry import build_model_from_config
from src.train import make_grad_scaler, train_one_epoch, validate
from src.utils import ensure_dir


def artifact_dir(name: str) -> Path:
    return ensure_dir(Path("outputs") / "test_artifacts" / f"{name}_{uuid4().hex}")


def test_dataset_missing_mask_error():
    root = artifact_dir("missing_mask")
    patient = root / "patient0001"
    patient.mkdir()
    Image.fromarray((torch.rand(32, 32).numpy() * 255).astype("uint8")).save(patient / "patient0001_2CH_ED.png")
    try:
        discover_camus_samples(root, require_masks=True)
    except FileNotFoundError as exc:
        assert "matching masks" in str(exc) or "No CAMUS-style" in str(exc)
    else:
        raise AssertionError("Expected missing-mask discovery to raise FileNotFoundError")


def test_dummy_dataset_and_one_epoch_training():
    data_root = create_dummy_camus_dataset(artifact_dir("dummy") / "data", num_patients=4, image_size=32)
    config = {
        "data_root": str(data_root),
        "image_size": 32,
        "num_classes": 4,
        "in_channels": 1,
        "batch_size": 2,
        "num_workers": 0,
        "seed": 7,
        "num_folds": 4,
        "fold": 0,
        "model": "baseline_unet",
        "model_params": {"base_channels": 4, "batch_norm": True, "dropout": 0.0},
        "loss": "dice_ce",
        "augmentation": {"enabled": False},
        "preprocessing": {"normalize": "minmax", "z_score": False, "class_mapping": None},
        "use_synthetic": False,
    }
    datasets = build_datasets(config, require_masks=True)
    assert len(datasets["train"]) > 0
    train_loader = DataLoader(datasets["train"], batch_size=2, shuffle=False, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=2, shuffle=False, num_workers=0)
    model = build_model_from_config(config)
    criterion = build_loss("dice_ce", num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(False)
    loss = train_one_epoch(model, train_loader, criterion, optimizer, torch.device("cpu"), scaler, use_amp=False)
    val_loss, metrics = validate(model, val_loader, criterion, torch.device("cpu"), num_classes=4)
    assert loss >= 0
    assert val_loss >= 0
    assert 0.0 <= metrics["mean_dice"] <= 1.0


def test_generic_image_mask_layout():
    root = artifact_dir("generic") / "data"
    images = root / "train" / "images"
    masks = root / "train" / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.fromarray((torch.rand(32, 32).numpy() * 255).astype("uint8")).save(images / "sample001.png")
    Image.fromarray(torch.randint(0, 4, (32, 32), dtype=torch.uint8).numpy()).save(masks / "sample001_mask.png")

    samples = discover_image_mask_samples(root, split="train", require_masks=True)
    assert len(samples) == 1
    config = {
        "data_root": str(root),
        "dataset_name": "generic",
        "image_size": 32,
        "num_classes": 4,
        "augmentation": {"enabled": False},
        "preprocessing": {"normalize": "minmax", "z_score": False, "class_mapping": None},
    }
    datasets = build_datasets(config, require_masks=True)
    image, mask, metadata = datasets["train"][0]
    assert image.shape == (1, 32, 32)
    assert mask.shape == (32, 32)
    assert metadata["patient_id"]
