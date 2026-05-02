# CAMUS 2D Echocardiography Segmentation

This repository implements a modern PyTorch research pipeline inspired by Leclerc et al., 2019, "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography." The original paper introduced and benchmarked segmentation methods on CAMUS for 2D echocardiography, targeting LV cavity, myocardium, left atrium, and background across 2CH/4CH ED/ES images.

This codebase keeps a simple baseline U-Net for fair reproduction-style comparison, then adds stronger augmentation, postprocessing, Attention U-Net, U-Net++, MultiResUNet, and optional temporal early-fusion U-Net.

## What Is Implemented

- Flexible CAMUS-style dataset discovery with patient folders, 2CH/4CH views, ED/ES phases, masks, metadata, quality labels, and pixel spacing when available.
- Single-frame and optional temporal-window inputs.
- Real, synthetic-only, real-only, and mixed real/synthetic training modes.
- Basic and ultrasound-specific augmentations: rotation, scale, crop, flip, brightness/contrast, Gaussian noise, speckle, gain, acoustic shadow dropout, elastic deformation, blur, and gamma.
- Losses: Dice, cross-entropy, Dice + CE, focal, Tversky, boundary, and temporal smoothness helper.
- Metrics: Dice, IoU, Hausdorff, HD95, average surface distance, approximate LVEDV/LVESV/LVEF, Bland-Altman plot.
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

For GPU training in Colab, use [COLAB.md](COLAB.md) and [notebooks/colab_train_test.ipynb](notebooks/colab_train_test.ipynb). The notebook mounts Google Drive, clones this repository, installs Colab-safe dependencies, runs tests, trains, evaluates, and saves checkpoints to Drive.

## Prepare CAMUS

Download CAMUS manually from the official challenge/dataset source, accept its license, and point `data_root` in `config.yaml` or pass `--data-root`.

Supported examples:

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

PNG/TIFF/NumPy/NIfTI-style layouts are also supported when filenames include patient, view (`2CH` or `4CH`), phase (`ED` or `ES`), and masks use tokens like `_gt`, `_mask`, `_seg`, or `_label`.

## Train

Baseline U-Net:

```powershell
python -m src.train --config config.yaml --data-root C:\path\to\CAMUS --model baseline_unet
```

Attention U-Net:

```powershell
python -m src.train --config config.yaml --data-root C:\path\to\CAMUS --model attention_unet
```

MultiResUNet:

```powershell
python -m src.train --config config.yaml --data-root C:\path\to\CAMUS --model multiresunet
```

Temporal early fusion:

```powershell
python -m src.train --config config.yaml --data-root C:\path\to\CAMUS --model temporal_unet --temporal-window 3
```

If neighboring frames are unavailable, temporal mode repeats the current frame so the code remains runnable; meaningful temporal gains require actual adjacent frames or sequence exports.
When CAMUS-style sequence files are present with names containing `sequence`, `seq`, `video`, `cine`, or `movie`, the dataset loader uses them to stack neighboring frames around the matching ED/ES still frame.

Smoke test without CAMUS:

```powershell
python -m src.train --config config.yaml --create-dummy-data --epochs 1 --batch-size 2 --image-size 64 --model baseline_unet
```

## Evaluate

```powershell
python -m src.evaluate --checkpoint outputs\run-name\best.pth --data-root C:\path\to\CAMUS --split test
```

Outputs include `metrics.csv`, `metrics.json`, `per_patient_metrics.csv`, qualitative overlays, failure cases, and Bland-Altman plots when clinical estimates are available.

## Predict

```powershell
python -m src.predict --checkpoint outputs\run-name\best.pth --input C:\path\to\patient0001 --output-dir outputs\predictions --save-confidence --postprocess
```

Prediction saves raw masks, color masks, overlays, optional softmax arrays, and `quality_flags.json`.

## Compare Experiments

```powershell
python -m src.compare_experiments outputs\baseline outputs\attention outputs\multires --output-dir outputs\comparison
```

This writes `comparison_report.csv`, summary JSON, training-curve plots, and a Dice-per-structure plot when class Dice metrics are available.

## Cross-Validation

Generate per-fold configs:

```powershell
python -m src.cross_validate --config config.yaml --output-dir outputs\cv --num-folds 5 --generate-configs
```

After training/evaluating folds, aggregate fold metrics:

```powershell
python -m src.cross_validate --output-dir outputs\cv
```

The aggregator looks for `metrics.json` under fold folders and writes `fold_metrics.csv`, `cross_validation_summary.csv`, and `cross_validation_summary.json`.

## Test

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
pytest -q tests
```

## Limitations

- This is research code, not a medical device.
- Results depend on exact CAMUS splits, preprocessing, and label conventions.
- Clinical volume estimates require correct view pairing, ED/ES masks, and spacing metadata.
- Simpson-style biplane estimates here are practical approximations, not validated clinical measurements.
- Temporal mode needs real neighboring frames to provide temporal information.
- Synthetic augmentation and synthetic data mixing must be validated carefully before claims are made.
