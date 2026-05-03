# Google Colab Training and Testing

Use Colab when you want free/hosted GPU training without configuring CUDA locally.

## 1. Put CAMUS in Google Drive

Recommended Drive layout:

```text
MyDrive/
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

You can also upload `CAMUS.zip` to `MyDrive` and unzip it from the notebook.

## 2. Open the Colab Notebook

Open [notebooks/colab_train_test.ipynb](notebooks/colab_train_test.ipynb) in Google Colab.

In Colab:

1. Runtime -> Change runtime type -> GPU.
2. Run the cells from top to bottom.
3. Set `DATA_ROOT` to your Drive CAMUS folder.
4. Start with a short smoke run before a long training job.

## 3. Useful Commands

After cloning the repo in Colab:

```bash
cd /content
git clone https://github.com/yasser2652/Deep-Echocardiographic-segmentation-.git /content/DeepEchoSeg
cd /content/DeepEchoSeg
PYTHONDONTWRITEBYTECODE=1 pytest -q tests
```

If Colab says `getcwd: cannot access parent directories`, run `cd /content` or restart the runtime before recloning. This happens when the notebook deletes the folder it was currently inside.

Train:

```bash
python -m src.train \
  --config config.yaml \
  --data-root /content/drive/MyDrive/CAMUS \
  --output-dir /content/drive/MyDrive/camus_outputs \
  --run-name baseline_unet_colab \
  --model baseline_unet \
  --epochs 100 \
  --batch-size 8 \
  --image-size 256 \
  --device cuda \
  --mixed-precision
```

Evaluate:

```bash
python -m src.evaluate \
  --checkpoint /content/drive/MyDrive/camus_outputs/baseline_unet_colab/best.pth \
  --data-root /content/drive/MyDrive/CAMUS \
  --output-dir /content/drive/MyDrive/camus_outputs/baseline_unet_colab/evaluation \
  --split test \
  --device cuda
```

## Notes

- Colab already includes PyTorch, so the notebook installs the medical-imaging and utility dependencies without reinstalling `torch`.
- Keep outputs in Google Drive so checkpoints survive runtime resets.
- CAMUS is not bundled with this repository. You must download it separately and follow its license.
