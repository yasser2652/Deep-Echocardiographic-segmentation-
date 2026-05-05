from pathlib import Path
from uuid import uuid4

import numpy as np

from src.dataset import build_datasets, create_dummy_camus_dataset
from src.postprocess import enforce_no_overlap_priority, postprocess_mask
from src.utils import ensure_dir


def artifact_dir(name: str) -> Path:
    return ensure_dir(Path("outputs") / "test_artifacts" / f"{name}_{uuid4().hex}")


def test_postprocessing_removes_small_components():
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    mask[2:4, 2:4] = 1
    processed = postprocess_mask(mask, num_classes=4, config={"min_region_size": 20, "keep_largest_component_per_class": True})
    assert processed[12:28, 12:28].sum() > 0
    assert not np.any(processed[2:4, 2:4] == 1)


def test_no_overlap_priority_prefers_lv_then_myo_then_la():
    shape = (16, 16)
    class_masks = {
        1: np.zeros(shape, dtype=bool),
        2: np.zeros(shape, dtype=bool),
        3: np.zeros(shape, dtype=bool),
    }
    class_masks[1][4:10, 4:10] = True
    class_masks[2][6:12, 6:12] = True
    class_masks[3][8:14, 8:14] = True
    out = enforce_no_overlap_priority(class_masks, shape, priority=[1, 2, 3])
    assert out[7, 7] == 1
    assert out[11, 11] == 2
    assert out[13, 13] == 3


def test_split_directory_prevents_patient_leakage():
    root = create_dummy_camus_dataset(artifact_dir("split") / "data", num_patients=4, image_size=32)
    split_dir = artifact_dir("splits")
    (split_dir / "train_patients.txt").write_text("patient0001\npatient0002\n", encoding="utf-8")
    (split_dir / "val_patients.txt").write_text("patient0003\n", encoding="utf-8")
    (split_dir / "test_patients.txt").write_text("patient0004\n", encoding="utf-8")
    config = {
        "data_root": str(root),
        "split_file": str(split_dir),
        "image_size": 32,
        "augmentation": {"enabled": False},
        "preprocessing": {"normalize": "minmax", "z_score": False, "class_mapping": None},
    }
    datasets = build_datasets(config, require_masks=True)
    split_patients = {
        split: {sample.patient_id for sample in dataset.samples}
        for split, dataset in datasets.items()
    }
    assert split_patients["train"] == {"patient0001", "patient0002"}
    assert split_patients["val"] == {"patient0003"}
    assert split_patients["test"] == {"patient0004"}
    assert split_patients["train"].isdisjoint(split_patients["val"])
    assert split_patients["train"].isdisjoint(split_patients["test"])
