# Future Codex Notes

This repository is a CPU-compatible PyTorch research pipeline for CAMUS-style 2D echocardiography segmentation.

## Run Checks

Use these from the repository root:

```powershell
python -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]"
$env:PYTHONDONTWRITEBYTECODE='1'; pytest -q tests
```

For a quick end-to-end smoke run without CAMUS:

```powershell
python -m src.train --config config.yaml --create-dummy-data --epochs 1 --batch-size 2 --image-size 64 --model baseline_unet --output-dir outputs/smoke
```

## Project Structure

- `src/dataset.py`: flexible CAMUS-style discovery, loading, metadata parsing, split construction, synthetic mixing, dummy dataset generation.
- `src/transforms.py`: resizing, normalization, basic and ultrasound-specific augmentation.
- `src/models/`: baseline U-Net, Attention U-Net, U-Net++, MultiResUNet, Temporal U-Net.
- `src/models/registry.py` and `src/model_registry.py`: model lookup by config/CLI name. Keep `src.model_registry` backward compatible.
- `src/models/gdkvm.py`, `src/models/echovim.py`, `src/models/osa.py`: research-inspired advanced echo segmentation models, not official reproductions unless explicitly updated.
- `src/losses.py`, `src/metrics.py`, `src/surface_metrics.py`: training and evaluation criteria.
- `src/clinical_metrics.py`: research-grade LVEDV/LVESV/LVEF estimates.
- `src/train.py`, `src/evaluate.py`, `src/predict.py`, `src/compare_experiments.py`, `src/cross_validate.py`: CLI workflows.
- `tests/`: model, loss, metric, clinical, dataset, and smoke-training tests.

## Coding Conventions

- Keep the baseline U-Net intact for fair comparisons.
- When adding a model, register it in `src/models/registry.py`, preserve the `src.model_registry` wrapper, and add a shape test.
- Do not hardcode local data paths. Use `config.yaml` or CLI overrides.
- Preserve CPU compatibility for all tests and smoke runs.
- Avoid loading full CAMUS data into memory; use `Dataset` and `DataLoader`.
- Keep patient-level splitting. Never split individual images from the same patient across train/val/test.
- Temporal mode should use real sequence/video frames when available and repeat ED/ES frames only as a fallback.
- Treat clinical metrics as research estimates. Do not present them as clinical measurements.
- Keep masks integer label maps: `0` background, `1` LV cavity, `2` myocardium, `3` left atrium.
- Advanced model tests should run on CPU with small `base_channels`.
