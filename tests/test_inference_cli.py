import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image

from src.model_registry import build_model_from_config
from src.utils import ensure_dir, save_checkpoint


def test_inference_cli_runs_with_dummy_checkpoint():
    root = ensure_dir(Path("outputs") / "test_artifacts" / f"inference_{uuid4().hex}")
    image_path = root / "image.png"
    output_path = root / "mask.png"
    ckpt_path = root / "best_dice.pt"
    Image.fromarray((torch.rand(32, 32).numpy() * 255).astype("uint8")).save(image_path)
    config = {
        "model": "gdkvm",
        "image_size": 32,
        "num_classes": 4,
        "in_channels": 1,
        "model_params": {"base_channels": 2, "batch_norm": True, "dropout": 0.0},
        "preprocessing": {"normalize": "minmax", "z_score": False},
    }
    model = build_model_from_config(config)
    save_checkpoint(ckpt_path, model, None, None, 1, config, {"mean_dice": 0.0})
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.inference",
            "--checkpoint",
            str(ckpt_path),
            "--image",
            str(image_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ],
        check=False,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()
