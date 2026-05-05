from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from skimage import draw
from torch.utils.data import Dataset

from src.transforms import SegmentationTransform
from src.utils import kfold_patient_split, read_split_file, split_patients


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz", ".mhd", ".nii", ".nii.gz")
MASK_TOKENS = ("_gt", "-gt", "mask", "seg", "label", "annotation")


@dataclass(frozen=True)
class CamusSample:
    image_path: Path
    mask_path: Path | None
    patient_id: str
    view: str
    phase: str
    sequence_path: Path | None = None
    spacing: tuple[float, float] | None = None
    quality: str | None = None
    frame_index: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ImageMaskSample:
    image_path: Path
    mask_path: Path | None
    patient_id: str
    view: str = "unknown"
    phase: str = "unknown"
    spacing: tuple[float, float] | None = None


def _lower_name(path: Path) -> str:
    return path.name.lower()


def is_supported_image(path: Path) -> bool:
    name = _lower_name(path)
    return any(name.endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_mask_path(path: Path) -> bool:
    name = _lower_name(path)
    return any(token in name for token in MASK_TOKENS)


def strip_medical_suffix(path: Path) -> str:
    name = path.name
    for ext in (".nii.gz", ".mhd", ".nii", ".tiff", ".tif", ".jpeg", ".jpg", ".png", ".bmp", ".npy", ".npz"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return path.stem


def infer_patient_id(path: Path) -> str:
    text = " ".join(part for part in path.parts)
    match = re.search(r"(patient[_-]?\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", "").replace("-", "").lower()
    for parent in [path.parent, *path.parents]:
        if parent.name.lower().startswith("patient"):
            return parent.name
    return path.parent.name


def infer_view(path: Path) -> str | None:
    text = strip_medical_suffix(path).lower().replace("-", "_")
    if re.search(r"(^|_)(2ch|2c)($|_)", text):
        return "2CH"
    if re.search(r"(^|_)(4ch|4c)($|_)", text):
        return "4CH"
    return None


def infer_phase(path: Path) -> str | None:
    text = strip_medical_suffix(path).lower().replace("-", "_")
    if re.search(r"(^|_)ed($|_)", text):
        return "ED"
    if re.search(r"(^|_)es($|_)", text):
        return "ES"
    return None


def parse_metadata_file(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.exists():
        return metadata
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, value = line.split(":", 1)
            elif "=" in line:
                key, value = line.split("=", 1)
            else:
                continue
            metadata[key.strip().lower()] = value.strip()
    return metadata


def patient_metadata(patient_dir: Path, view: str | None = None) -> dict[str, str]:
    candidates = []
    if view:
        candidates.extend(patient_dir.glob(f"*{view}*.cfg"))
        candidates.extend(patient_dir.glob(f"*{view.lower()}*.cfg"))
    candidates.extend(patient_dir.glob("*.cfg"))
    merged: dict[str, str] = {}
    for file in candidates:
        merged.update(parse_metadata_file(file))
    return merged


def _parse_float_pair(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        v = float(nums[0])
        return v, v
    return None


def spacing_from_metadata(metadata: dict[str, str]) -> tuple[float, float] | None:
    for key in ("pixelspacing", "pixel spacing", "spacing", "resolution", "image spacing"):
        spacing = _parse_float_pair(metadata.get(key))
        if spacing is not None:
            return spacing
    x = metadata.get("spacing_x") or metadata.get("resolution_x")
    y = metadata.get("spacing_y") or metadata.get("resolution_y")
    if x and y:
        try:
            return float(y), float(x)
        except ValueError:
            return None
    return None


def quality_from_metadata(metadata: dict[str, str]) -> str | None:
    for key in ("imagequality", "image quality", "quality"):
        if key in metadata:
            return metadata[key]
    return None


def frame_index_from_metadata(metadata: dict[str, str], phase: str) -> int | None:
    for key in (phase.lower(), f"{phase.lower()}frame", f"{phase.lower()} frame"):
        if key in metadata:
            nums = re.findall(r"\d+", metadata[key])
            if nums:
                return int(nums[0])
    return None


def load_medical_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    name = _lower_name(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if name.endswith(".npy"):
        return np.asarray(np.load(path))
    if name.endswith(".npz"):
        data = np.load(path)
        key = data.files[0]
        return np.asarray(data[key])
    if name.endswith((".mhd", ".nii", ".nii.gz")):
        try:
            import SimpleITK as sitk

            image = sitk.ReadImage(str(path))
            return sitk.GetArrayFromImage(image)
        except Exception:
            if name.endswith((".nii", ".nii.gz")):
                import nibabel as nib

                return np.asarray(nib.load(str(path)).get_fdata())
            raise
    with Image.open(path) as img:
        array = np.asarray(img.convert("L") if img.mode not in ("I", "F") else img)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = array[..., :3].mean(axis=-1)
    return np.asarray(array)


def select_2d(array: np.ndarray, frame_index: int | None = None) -> np.ndarray:
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[-1] in (3, 4):
            return array[..., :3].mean(axis=-1)
        idx = frame_index if frame_index is not None else array.shape[0] // 2
        idx = int(np.clip(idx, 0, array.shape[0] - 1))
        return array[idx]
    raise ValueError(f"Could not reduce array with shape {array.shape} to a 2D frame.")


def find_matching_mask(image_path: Path, masks: list[Path]) -> Path | None:
    base = strip_medical_suffix(image_path)
    suffixes = [image_path.name[len(base) :]]
    exact_names = []
    for suffix in suffixes:
        exact_names.extend(
            [
                f"{base}_gt{suffix}",
                f"{base}-gt{suffix}",
                f"{base}_mask{suffix}",
                f"{base}_seg{suffix}",
                f"{base}_label{suffix}",
            ]
        )
    for name in exact_names:
        candidate = image_path.with_name(name)
        if candidate.exists():
            return candidate

    patient = infer_patient_id(image_path)
    view = infer_view(image_path)
    phase = infer_phase(image_path)
    candidates = []
    for mask in masks:
        if infer_patient_id(mask) != patient:
            continue
        if view and infer_view(mask) != view:
            continue
        if phase and infer_phase(mask) != phase:
            continue
        candidates.append(mask)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.parent != image_path.parent, len(p.name)))
    return candidates[0]


def _pair_key(path: Path) -> str:
    key = strip_medical_suffix(path).lower()
    for token in MASK_TOKENS:
        key = key.replace(token, "")
    return key.strip("_- .")


def discover_image_mask_samples(
    root: str | Path,
    split: str | None = None,
    require_masks: bool = True,
) -> list[ImageMaskSample]:
    """Discover generic image/mask pairs.

    Supported layouts include:
    data/train/images/*.png and data/train/masks/*.png, or a single folder with
    image files and masks named with *_mask, *_gt, *_seg, or *_label suffixes.
    """

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")
    split_root = root / split if split and (root / split).exists() else root
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    if images_dir.exists():
        image_files = [p for p in images_dir.rglob("*") if p.is_file() and is_supported_image(p) and not is_mask_path(p)]
        mask_files = [p for p in masks_dir.rglob("*") if p.is_file() and is_supported_image(p)] if masks_dir.exists() else []
    else:
        files = [p for p in split_root.rglob("*") if p.is_file() and is_supported_image(p)]
        mask_files = [p for p in files if is_mask_path(p)]
        image_files = [p for p in files if not is_mask_path(p)]

    mask_by_key = {_pair_key(mask): mask for mask in sorted(mask_files)}
    samples: list[ImageMaskSample] = []
    missing: list[Path] = []
    for image_path in sorted(image_files):
        mask_path = mask_by_key.get(_pair_key(image_path))
        if mask_path is None:
            mask_path = find_matching_mask(image_path, mask_files)
        if require_masks and mask_path is None:
            missing.append(image_path)
            continue
        samples.append(
            ImageMaskSample(
                image_path=image_path,
                mask_path=mask_path,
                patient_id=infer_patient_id(image_path),
                view=infer_view(image_path) or "unknown",
                phase=infer_phase(image_path) or "unknown",
                spacing=(1.0, 1.0),
            )
        )

    if require_masks and not samples and missing:
        examples = "\n".join(str(p) for p in missing[:5])
        raise FileNotFoundError(
            "Found images but no matching masks. Expected masks in a sibling `masks/` folder or names like "
            "`*_mask.*`, `*_gt.*`, `*_seg.*`, or `*_label.*`.\n"
            f"First unmatched images:\n{examples}"
        )
    if not samples:
        target = split_root if split is None else f"{root}/{split}"
        raise FileNotFoundError(f"No generic image/mask samples found under {target}.")
    return samples


def is_sequence_path(path: Path) -> bool:
    name = strip_medical_suffix(path).lower()
    if is_mask_path(path):
        return False
    return any(token in name for token in ("sequence", "seq", "video", "cine", "movie"))


def find_matching_sequence(image_path: Path, sequence_files: list[Path]) -> Path | None:
    patient = infer_patient_id(image_path)
    view = infer_view(image_path)
    candidates = []
    for sequence_path in sequence_files:
        if infer_patient_id(sequence_path) != patient:
            continue
        if view and infer_view(sequence_path) != view:
            continue
        candidates.append(sequence_path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.parent != image_path.parent, len(p.name)))
    return candidates[0]


def discover_camus_samples(
    root: str | Path,
    require_masks: bool = True,
    patient_ids: Iterable[str] | None = None,
) -> list[CamusSample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"CAMUS data root not found: {root}")
    patient_filter = {pid.lower() for pid in patient_ids} if patient_ids is not None else None
    files = [p for p in root.rglob("*") if p.is_file() and is_supported_image(p)]
    masks = [p for p in files if is_mask_path(p)]
    sequences = [p for p in files if is_sequence_path(p)]
    images = [p for p in files if not is_mask_path(p) and not is_sequence_path(p)]
    samples: list[CamusSample] = []
    missing_masks: list[Path] = []

    for image_path in sorted(images):
        view = infer_view(image_path)
        phase = infer_phase(image_path)
        if view is None or phase is None:
            continue
        patient_id = infer_patient_id(image_path)
        if patient_filter is not None and patient_id.lower() not in patient_filter:
            continue
        metadata = patient_metadata(image_path.parent, view)
        spacing = spacing_from_metadata(metadata)
        quality = quality_from_metadata(metadata)
        frame_idx = frame_index_from_metadata(metadata, phase)
        mask_path = find_matching_mask(image_path, masks)
        sequence_path = find_matching_sequence(image_path, sequences)
        if require_masks and mask_path is None:
            missing_masks.append(image_path)
            continue
        samples.append(
            CamusSample(
                image_path=image_path,
                mask_path=mask_path,
                patient_id=patient_id,
                view=view,
                phase=phase,
                sequence_path=sequence_path,
                spacing=spacing,
                quality=quality,
                frame_index=frame_idx,
                metadata=metadata,
            )
        )

    if require_masks and not samples and missing_masks:
        examples = "\n".join(str(p) for p in missing_masks[:5])
        raise FileNotFoundError(
            "Found CAMUS-like image files but no matching masks. Expected names such as "
            "`patient0001_2CH_ED_gt.*`, `*_mask.*`, or `*_seg.*` next to the image.\n"
            f"First unmatched images:\n{examples}"
        )
    if not samples:
        raise FileNotFoundError(
            f"No CAMUS-style samples found under {root}. Expected patient folders containing "
            "2CH/4CH and ED/ES image files, with masks for training/evaluation."
        )
    return samples


class CamusDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        samples: list[CamusSample] | None = None,
        transform: SegmentationTransform | None = None,
        image_size: int | tuple[int, int] = 256,
        training: bool = False,
        require_masks: bool = True,
        class_mapping: dict[int, int] | None = None,
        temporal_window: int = 1,
    ) -> None:
        self.root = Path(root)
        self.samples = samples if samples is not None else discover_camus_samples(root, require_masks=require_masks)
        self.transform = transform or SegmentationTransform(image_size=image_size, training=training)
        self.require_masks = require_masks
        self.class_mapping = class_mapping
        self.temporal_window = max(1, int(temporal_window))

    def __len__(self) -> int:
        return len(self.samples)

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.int64, copy=False)
        if not self.class_mapping:
            return mask
        out = np.zeros_like(mask, dtype=np.int64)
        for source, target in self.class_mapping.items():
            out[mask == int(source)] = int(target)
        return out

    def _load_frame(self, sample: CamusSample) -> np.ndarray:
        return select_2d(load_medical_image(sample.image_path), sample.frame_index)

    @staticmethod
    def _frame_at(array: np.ndarray, index: int) -> np.ndarray:
        array = np.asarray(array)
        array = np.squeeze(array)
        if array.ndim == 2:
            return array
        if array.ndim == 3 and array.shape[-1] in (3, 4):
            return array[..., :3].mean(axis=-1)
        if array.ndim != 3:
            raise ValueError(f"Expected 2D or 3D temporal image array, got shape {array.shape}")
        index = int(np.clip(index, 0, array.shape[0] - 1))
        return array[index]

    def _temporal_indices(self, center: int, num_frames: int) -> list[int]:
        half = self.temporal_window // 2
        if self.temporal_window % 2 == 1:
            offsets = list(range(-half, half + 1))
        else:
            offsets = list(range(-half, half))
        return [int(np.clip(center + offset, 0, num_frames - 1)) for offset in offsets[: self.temporal_window]]

    def _infer_sequence_center(self, sequence: np.ndarray, current: np.ndarray, sample: CamusSample) -> int:
        if sequence.ndim != 3 or sequence.shape[-1] in (3, 4):
            return 0
        current_2d = np.asarray(current)
        if current_2d.shape == sequence.shape[-2:]:
            distances = [float(np.mean((self._frame_at(sequence, idx).astype(np.float32) - current_2d.astype(np.float32)) ** 2)) for idx in range(sequence.shape[0])]
            return int(np.argmin(distances))
        if sample.frame_index is not None:
            return int(np.clip(sample.frame_index, 0, sequence.shape[0] - 1))
        return sequence.shape[0] // 2

    def _temporal_stack(self, sample: CamusSample) -> np.ndarray:
        current = self._load_frame(sample)
        if self.temporal_window <= 1:
            return current
        if sample.sequence_path is not None:
            sequence = np.squeeze(load_medical_image(sample.sequence_path))
            if sequence.ndim == 3 and sequence.shape[-1] not in (3, 4):
                center = self._infer_sequence_center(sequence, current, sample)
                indices = self._temporal_indices(center, sequence.shape[0])
                return np.stack([self._frame_at(sequence, idx) for idx in indices], axis=0)
        # CAMUS ED/ES files are often single-frame exports. If neighboring video frames are not
        # discoverable, repeating the current frame preserves the configured channel contract.
        return np.stack([current for _ in range(self.temporal_window)], axis=0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        image = self._temporal_stack(sample)
        if sample.mask_path is None:
            if self.require_masks:
                raise FileNotFoundError(f"Missing segmentation mask for image: {sample.image_path}")
            mask = np.zeros(select_2d(image).shape[-2:], dtype=np.int64)
        else:
            mask = select_2d(load_medical_image(sample.mask_path), sample.frame_index)
            mask = self._remap_mask(mask)
        image_tensor, mask_tensor = self.transform(image, mask)
        assert mask_tensor is not None
        metadata = {
            "patient_id": sample.patient_id,
            "view": sample.view,
            "phase": sample.phase,
            "spacing": sample.spacing if sample.spacing is not None else (1.0, 1.0),
            "spacing_available": sample.spacing is not None,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path) if sample.mask_path is not None else None,
            "sequence_path": str(sample.sequence_path) if sample.sequence_path is not None else None,
            "quality": sample.quality,
        }
        return image_tensor, mask_tensor, metadata


class ImageMaskDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        samples: list[ImageMaskSample] | None = None,
        transform: SegmentationTransform | None = None,
        image_size: int | tuple[int, int] = 256,
        training: bool = False,
        require_masks: bool = True,
        class_mapping: dict[int, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.samples = samples if samples is not None else discover_image_mask_samples(root, split=split, require_masks=require_masks)
        self.transform = transform or SegmentationTransform(image_size=image_size, training=training)
        self.require_masks = require_masks
        self.class_mapping = class_mapping

    def __len__(self) -> int:
        return len(self.samples)

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.int64, copy=False)
        if not self.class_mapping:
            return mask
        out = np.zeros_like(mask, dtype=np.int64)
        for source, target in self.class_mapping.items():
            out[mask == int(source)] = int(target)
        return out

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        image = select_2d(load_medical_image(sample.image_path))
        if sample.mask_path is None:
            if self.require_masks:
                raise FileNotFoundError(f"Missing segmentation mask for image: {sample.image_path}")
            mask = np.zeros(image.shape[-2:], dtype=np.int64)
        else:
            mask = self._remap_mask(select_2d(load_medical_image(sample.mask_path)))
        image_tensor, mask_tensor = self.transform(image, mask)
        assert mask_tensor is not None
        metadata = {
            "patient_id": sample.patient_id,
            "view": sample.view,
            "phase": sample.phase,
            "spacing": sample.spacing if sample.spacing is not None else (1.0, 1.0),
            "spacing_available": sample.spacing is not None,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path) if sample.mask_path is not None else None,
            "sequence_path": None,
            "quality": None,
        }
        return image_tensor, mask_tensor, metadata


class SyntheticMixDataset(Dataset):
    def __init__(
        self,
        real_dataset: Dataset,
        synthetic_dataset: Dataset | None,
        synthetic_ratio: float = 0.25,
        mode: str = "mixed_real_synthetic",
        seed: int = 42,
    ) -> None:
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.synthetic_ratio = float(np.clip(synthetic_ratio, 0.0, 1.0))
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        if mode == "synthetic_only" and synthetic_dataset is None:
            raise ValueError("synthetic_only mode requires synthetic_dataset.")

    def __len__(self) -> int:
        if self.mode == "synthetic_only":
            return len(self.synthetic_dataset)  # type: ignore[arg-type]
        if self.synthetic_dataset is None or self.mode == "real_only":
            return len(self.real_dataset)
        return max(len(self.real_dataset), len(self.synthetic_dataset))

    def __getitem__(self, index: int):
        if self.mode == "synthetic_only":
            return self.synthetic_dataset[index % len(self.synthetic_dataset)]  # type: ignore[index]
        use_synth = (
            self.synthetic_dataset is not None
            and self.mode == "mixed_real_synthetic"
            and self.rng.random() < self.synthetic_ratio
        )
        if use_synth:
            return self.synthetic_dataset[index % len(self.synthetic_dataset)]  # type: ignore[index]
        return self.real_dataset[index % len(self.real_dataset)]


def make_split_samples(samples: list[CamusSample], config: dict[str, Any]) -> dict[str, list[CamusSample]]:
    patient_ids = sorted({s.patient_id for s in samples})
    if config.get("split_file"):
        split_ids = read_split_file(config["split_file"])
    elif config.get("num_folds", 0):
        split_ids = kfold_patient_split(
            patient_ids,
            fold=int(config.get("fold", 0)),
            num_folds=int(config.get("num_folds", 5)),
            seed=int(config.get("seed", 42)),
        )
    else:
        split_ids = split_patients(
            patient_ids,
            seed=int(config.get("seed", 42)),
            train_fraction=float(config.get("train_fraction", 0.8)),
            val_fraction=float(config.get("val_fraction", 0.1)),
        )
    normalized = {split: {pid.lower() for pid in ids} for split, ids in split_ids.items()}
    return {split: [sample for sample in samples if sample.patient_id.lower() in ids] for split, ids in normalized.items()}


def make_split_image_samples(samples: list[ImageMaskSample], config: dict[str, Any]) -> dict[str, list[ImageMaskSample]]:
    patient_ids = sorted({s.patient_id for s in samples})
    if config.get("split_file"):
        split_ids = read_split_file(config["split_file"])
    else:
        split_ids = split_patients(
            patient_ids,
            seed=int(config.get("seed", 42)),
            train_fraction=float(config.get("train_fraction", 0.8)),
            val_fraction=float(config.get("val_fraction", 0.1)),
        )
    normalized = {split: {pid.lower() for pid in ids} for split, ids in split_ids.items()}
    return {split: [sample for sample in samples if sample.patient_id.lower() in ids] for split, ids in normalized.items()}


def has_generic_split_layout(root: str | Path) -> bool:
    root = Path(root)
    if (root / "images").exists() and (root / "masks").exists():
        return True
    return any((root / split / "images").exists() and (root / split / "masks").exists() for split in ("train", "val", "test"))


def build_generic_datasets(config: dict[str, Any], require_masks: bool = True) -> dict[str, Dataset]:
    data_root = Path(config["data_root"])
    prep = config.get("preprocessing", {}) or {}
    aug = config.get("augmentation", {}) or {}
    image_size = int(config.get("image_size", 256))
    class_mapping = prep.get("class_mapping")
    datasets: dict[str, Dataset] = {}
    split_dirs = [split for split in ("train", "val", "test") if (data_root / split / "images").exists()]
    if split_dirs:
        for split_name in split_dirs:
            transform = SegmentationTransform(
                image_size=image_size,
                training=split_name == "train",
                augmentation=aug,
                normalize=prep.get("normalize", "minmax"),
                z_score=bool(prep.get("z_score", False)),
            )
            datasets[split_name] = ImageMaskDataset(
                data_root,
                split=split_name,
                transform=transform,
                training=split_name == "train",
                require_masks=require_masks,
                class_mapping=class_mapping,
            )
        if "val" not in datasets and "test" in datasets:
            datasets["val"] = datasets["test"]
        if "test" not in datasets and "val" in datasets:
            datasets["test"] = datasets["val"]
        return datasets

    all_samples = discover_image_mask_samples(data_root, require_masks=require_masks)
    splits = make_split_image_samples(all_samples, config)
    for split_name, split_samples in splits.items():
        transform = SegmentationTransform(
            image_size=image_size,
            training=split_name == "train",
            augmentation=aug,
            normalize=prep.get("normalize", "minmax"),
            z_score=bool(prep.get("z_score", False)),
        )
        datasets[split_name] = ImageMaskDataset(
            data_root,
            samples=split_samples,
            transform=transform,
            training=split_name == "train",
            require_masks=require_masks,
            class_mapping=class_mapping,
        )
    return datasets


def build_datasets(config: dict[str, Any], require_masks: bool = True) -> dict[str, Dataset]:
    data_root = config.get("data_root")
    if not data_root:
        raise ValueError("data_root is not set. Point it to a CAMUS root or use the dummy test generator.")
    dataset_name = str(config.get("dataset_name", "camus")).lower()
    if dataset_name in {"generic", "image_mask", "echonet_dynamic"} or has_generic_split_layout(data_root):
        return build_generic_datasets(config, require_masks=require_masks)
    all_samples = discover_camus_samples(data_root, require_masks=require_masks)
    splits = make_split_samples(all_samples, config)
    prep = config.get("preprocessing", {}) or {}
    aug = config.get("augmentation", {}) or {}
    image_size = int(config.get("image_size", 256))
    temporal_window = int(config.get("temporal_window", 1))
    class_mapping = prep.get("class_mapping")
    datasets: dict[str, Dataset] = {}
    for split_name, split_samples in splits.items():
        transform = SegmentationTransform(
            image_size=image_size,
            training=split_name == "train",
            augmentation=aug,
            normalize=prep.get("normalize", "minmax"),
            z_score=bool(prep.get("z_score", False)),
        )
        datasets[split_name] = CamusDataset(
            data_root,
            samples=split_samples,
            transform=transform,
            training=split_name == "train",
            require_masks=require_masks,
            class_mapping=class_mapping,
            temporal_window=temporal_window,
        )
    if config.get("use_synthetic") and config.get("synthetic_data_root"):
        synth_samples = discover_camus_samples(config["synthetic_data_root"], require_masks=True)
        synth_transform = SegmentationTransform(
            image_size=image_size,
            training=True,
            augmentation=aug,
            normalize=prep.get("normalize", "minmax"),
            z_score=bool(prep.get("z_score", False)),
        )
        synth_dataset = CamusDataset(
            config["synthetic_data_root"],
            samples=synth_samples,
            transform=synth_transform,
            training=True,
            require_masks=True,
            temporal_window=temporal_window,
        )
        datasets["train"] = SyntheticMixDataset(
            datasets["train"],
            synth_dataset,
            synthetic_ratio=float(config.get("synthetic_ratio", 0.25)),
            mode=str(config.get("synthetic_mode", "mixed_real_synthetic")),
            seed=int(config.get("seed", 42)),
        )
    return datasets


def create_dummy_camus_dataset(root: str | Path, num_patients: int = 6, image_size: int = 96, seed: int = 42) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    for patient_idx in range(1, num_patients + 1):
        patient_id = f"patient{patient_idx:04d}"
        patient_dir = root / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        for view in ("2CH", "4CH"):
            info = patient_dir / f"Info_{view}.cfg"
            info.write_text("ED: 1\nES: 2\nImageQuality: Good\nPixelSpacing: 1.0 1.0\n", encoding="utf-8")
            sequence_frames = []
            for phase in ("ED", "ES"):
                mask = np.zeros((image_size, image_size), dtype=np.uint8)
                cx = image_size * (0.50 + rng.normal(0, 0.025))
                cy = image_size * (0.56 + rng.normal(0, 0.025))
                lv_r_y = image_size * (0.24 if phase == "ED" else 0.18)
                lv_r_x = image_size * (0.16 if phase == "ED" else 0.12)
                rr, cc = draw.ellipse(cy, cx, lv_r_y * 1.35, lv_r_x * 1.35, shape=mask.shape)
                mask[rr, cc] = 2
                rr, cc = draw.ellipse(cy, cx, lv_r_y, lv_r_x, shape=mask.shape)
                mask[rr, cc] = 1
                rr, cc = draw.ellipse(cy - image_size * 0.27, cx, image_size * 0.12, image_size * 0.18, shape=mask.shape)
                mask[rr, cc] = 3
                image = 0.25 + 0.35 * np.exp(-(((yy - cy) ** 2) / (2 * (image_size * 0.26) ** 2)))
                image += 0.35 * (mask > 0).astype(np.float32)
                image += rng.normal(0, 0.08, size=image.shape)
                image = np.clip(image, 0.0, 1.0)
                image_u8 = (image * 255).astype(np.uint8)
                stem = f"{patient_id}_{view}_{phase}"
                Image.fromarray(image_u8).save(patient_dir / f"{stem}.png")
                Image.fromarray(mask).save(patient_dir / f"{stem}_gt.png")
                sequence_frames.append(image_u8)
            if sequence_frames:
                middle = ((sequence_frames[0].astype(np.float32) + sequence_frames[-1].astype(np.float32)) / 2).astype(np.uint8)
                np.save(patient_dir / f"{patient_id}_{view}_sequence.npy", np.stack([sequence_frames[0], middle, sequence_frames[-1]], axis=0))
    return root
