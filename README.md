# CAMUS 2D Echocardiography Segmentation

This repository implements a modular PyTorch research pipeline inspired by Leclerc et al., 2019, "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography." The main CAMUS task is segmentation of background, LV cavity, myocardium, and left atrium in 2CH/4CH ED/ES echocardiography frames.

The project keeps a simple baseline U-Net for fair comparison, then adds stronger augmentation, postprocessing, Attention U-Net, U-Net++, MultiResUNet, temporal early-fusion U-Net, and three advanced repo-compatible model families:

- `gdkvm`: dynamic key-value-memory-inspired encoder-decoder segmentation.
- `echovim`: Vision Mamba/state-space-inspired dense segmentation.
- `osa`: orthogonal state-update and anatomical-prior-aware segmentation.

The `gdkvm`, `echovim`, and `osa` implementations are practical PyTorch research-inspired approximations built to work in this repository. They are not claimed to be official reproductions unless an official implementation is explicitly integrated later.

## What Is Implemented

- Flexible CAMUS-style dataset discovery with patient folders, 2CH/4CH views, ED/ES phases, masks, metadata, quality labels, and pixel spacing when available.
- Generic image/mask segmentation layout support for datasets such as curated EchoNet-Dynamic frame exports.
- Single-frame inputs and optional temporal/video inputs where models support them.
- Real, synthetic-only, real-only, and mixed real/synthetic training modes.
- Basic and ultrasound-specific augmentations: mild rotation, scale, translation, brightness/contrast, Gaussian noise, speckle, gain, acoustic shadow dropout, mild elastic deformation, blur, and gamma.
- Losses: Dice, cross-entropy, Dice + CE, focal, Tversky, boundary, temporal smoothness helper, and `combined_clinical`.
- Metrics: Dice, IoU, Hausdorff, HD95, average surface distance, CAMUS-style biplane Simpson LVEDV/LVESV/LVEF estimates, and Bland-Altman plot.
- Prediction QC checks for empty masks, disconnected LV cavity, abnormal areas, low confidence, and anatomy warnings.
- K-fold patient-level cross-validation support.
- Dummy CAMUS-like dataset generator for tests and smoke runs.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Google Colab

For GPU training in Colab, use [COLAB.md](COLAB.md) and [notebooks/colab_train_test.ipynb](notebooks/colab_train_test.ipynb). The notebook mounts Google Drive, clones this repository, installs Colab-safe dependencies, optionally copies CAMUS from Drive to local Colab disk, runs tests, trains, evaluates, and saves checkpoints to Drive.

## Dataset Formats

### CAMUS Layout

Download CAMUS manually from the official challenge/dataset source, accept its license, and point `data_root` in `config.yaml` or pass `--data-root`.

```text
CAMUS/
  patient0001/
    patient0001_2CH_ED.mhd
    patient0001_2CH_ED_gt.mhd
    patient0001_2CH_ES.mhd
    patient0001_2CH_ES_gt.mhd
    patient0001_4CH_ED.mhd
    patient0001_4CH_ED_gt.mhd
    Info_2CH.cfg
    Info_4CH.cfg
```

PNG/TIFF/NumPy/NIfTI layouts are also supported when filenames include patient, view (`2CH` or `4CH`), phase (`ED` or `ES`), and masks use tokens like `_gt`, `_mask`, `_seg`, or `_label`.

### Generic Image/Mask Layout

Use `dataset_name: generic` for a simple segmentation dataset:

```text
data/
  train/
    images/
      sample001.png
    masks/
      sample001_mask.png
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Supported file types include `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.npy`, `.npz`, `.nii`, and `.nii.gz` when medical loading dependencies are installed.

## Available Models

The model registry accepts:

```text
baseline_unet, unet, attention_unet, unetpp, unet++, multiresunet,
temporal_unet, gdkvm, echovim, echo_vim, osa,
resnet18_unet, resnet34_unet, resnet50_unet
```

The baseline U-Net is intentionally preserved so experiments can compare:

- baseline U-Net
- baseline U-Net plus stronger augmentation
- baseline U-Net plus postprocessing
- Attention U-Net
- U-Net++
- MultiResUNet
- Temporal U-Net
- GDKVM-inspired model
- EchoVim-inspired model
- OSA-inspired model
- ResNet encoder U-Net with resnet18, resnet34, or resnet50 encoder

## Train

Baseline U-Net:

```powershell
python train.py --config config.yaml --data-root C:\path\to\CAMUS --model unet
```

GDKVM:

```powershell
python train.py --config config.yaml --data-root C:\path\to\CAMUS --model gdkvm
```

EchoVim:

```powershell
python train.py --config config.yaml --data-root C:\path\to\CAMUS --model echovim
```

OSA:

```powershell
python train.py --config config.yaml --data-root C:\path\to\CAMUS --model osa
```

Attention U-Net and MultiResUNet:

```powershell
python train.py --config config.yaml --data-root C:\path\to\CAMUS --model attention_unet
python train.py --config config.yaml --data-root C:\path\to\CAMUS --model multiresunet
```

Temporal early fusion:

```powershell
python train.py --config config.yaml --data-root C:\path\to\CAMUS --model temporal_unet --temporal-window 3
```

If neighboring frames are unavailable, temporal mode repeats the current frame so the code remains runnable. Meaningful temporal gains require actual adjacent frames or sequence exports. CAMUS-style sequence files with names containing `sequence`, `seq`, `video`, `cine`, or `movie` are used when available.

Smoke test without CAMUS:

```powershell
python train.py --config config.yaml --create-dummy-data --epochs 1 --batch-size 2 --num-workers 0 --image-size 64 --model baseline_unet
```

Training writes `latest.pth`, `best.pth`, `last.pt`, `best_dice.pt`, `training_log.csv`, `training_summary.json`, a resolved config, and validation overlays.

Combined clinical segmentation loss:

```powershell
python train.py --config config.yaml --data-root C:\path\to\CAMUS --model unet --loss combined_clinical --class-weights 0.2,1.0,1.0,1.0
```

Patient split files are supported with `--split-file splits`, where `splits/` contains `train_patients.txt`, `val_patients.txt`, and `test_patients.txt`.

## Evaluate

```powershell
python evaluate.py --model gdkvm --checkpoint outputs\run-name\best_dice.pt --data-dir C:\path\to\CAMUS --split test --postprocess --camus-clinical-metrics --save-json --save-csv
```

Outputs include `metrics.csv`, `metrics.json`, `metrics_by_view_phase.csv/json`, `per_patient_metrics.csv`, `patient_clinical_metrics.csv/json`, qualitative overlays, failure cases, and Bland-Altman plots when clinical estimates are available.

## Predict A Folder

```powershell
python -m src.predict --checkpoint outputs\run-name\best_dice.pt --input C:\path\to\patient0001 --output-dir outputs\predictions --save-confidence --postprocess
```

Prediction saves raw masks, color masks, overlays, optional softmax arrays, and `quality_flags.json`.

## Single-Image Inference

```powershell
python inference.py ^
  --model osa ^
  --checkpoint outputs\osa_run\best_dice.pt ^
  --image C:\path\to\image.png ^
  --output outputs\prediction.png ^
  --overlay outputs\prediction_overlay.png
```

## Compare Trained Models

```powershell
python compare_models.py ^
  --models unet gdkvm echovim osa ^
  --checkpoints outputs ^
  --data-dir C:\path\to\CAMUS ^
  --output outputs\model_comparison.csv
```

This writes a table with model name, Dice, IoU, trainable parameter count, FPS, and checkpoint path.

Existing experiment-folder comparison is still available:

```powershell
python -m src.compare_experiments outputs\baseline outputs\attention outputs\multires --output-dir outputs\comparison
```

## Cross-Validation

Generate per-fold configs:

```powershell
python -m src.cross_validate --config config.yaml --output-dir outputs\cv --num-folds 5 --generate-configs
```

After training/evaluating folds, aggregate fold metrics:

```powershell
python -m src.cross_validate --output-dir outputs\cv
```

## Test

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
pytest -q tests
```

Quick syntax check:

```powershell
python -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]"
```

## References

- Leclerc et al., 2019, "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography."
- CAMUS dataset/challenge documentation from the official CAMUS source.
- GDKVM public project page: https://github.com/wangrui2025/GDKVM
- EchoVim and OSA architecture references should be filled with the exact paper/code links used in a future official integration.

## Limitations

- This is research code, not a medical device.
- GDKVM, EchoVim, and OSA are repo-compatible approximations, not official reproductions.
- Results depend on exact CAMUS splits, preprocessing, and label conventions.
- Clinical volume estimates require correct view pairing, ED/ES masks, and spacing metadata.
- Simpson-style biplane estimates here are practical approximations, not validated clinical measurements.
- Temporal mode needs real neighboring frames to provide temporal information.
- EchoNet-Dynamic segmentation support assumes you export frames and masks into the generic image/mask layout.
- Synthetic augmentation and synthetic data mixing must be validated carefully before claims are made.
