# ViT-UNet SAR Oil Spill Segmentation

Code and reproducibility artifacts for the IEEE Access paper on SAR oil-spill
segmentation with a ViT-UNet model.

The historical code id `vit_unet_4skip` is kept for checkpoint/script
compatibility. Architecturally, the model uses lateral skip features from ViT
blocks 2, 5, and 8; block 11 is the bottleneck feature passed into the decoder.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Layout

Download the datasets separately. They are intentionally not stored in git.

```text
dataset/
  images/images/train/*.png
  images/images/val/*.png
  masks/masks/train/*.png
  masks/masks/val/*.png

dataset_orig/
  train/{palsar,sentinel}/{image,label}/*.png
  test/{palsar,sentinel}/{image,label}/*.png
```

Model checkpoints (`*.pth`) are also not stored in git. Put them in
`models_results/` and `models_results_orig/` if you want to recompute metrics
from weights.

## Reproduce Tables

Recompute metrics from checkpoints:

```bash
python scripts/recompute_all_metrics.py --dataset refined
python scripts/recompute_all_metrics.py --dataset original
python scripts/recompute_all_metrics.py --ablation
python scripts/compute_model_complexity.py
```

Main CSV outputs:

```text
models_results/recomputed_metrics.csv
models_results/all_models_mean_std.csv
models_results/detailed_ablation.csv
models_results/model_complexity_analysis.csv
models_results_orig/recomputed_metrics.csv
models_results_orig/all_models_mean_std.csv
```

## Regenerate Figures

The quantitative figures are generated from the CSV files already included in
the repository:

```bash
python scripts/create_all_figures_master.py
```

Dataset and qualitative figures require the datasets and checkpoints:

```bash
python scripts/create_figure1_dataset.py
python scripts/create_qualitative_all_models.py
```

Generated figures are written to `figures/`.

## Train

Train the proposed model on the refined dataset:

```bash
python scripts/train.py --model vit_unet_4skip --dataset refined --data_path ./dataset --save_dir ./models_results --tag vit_unet_4skip --epochs 100 --early_stop_patience 15 --pos_weight 1.5 --seed 42
```

Train the same configuration on the original Deep-SAR layout:

```bash
python scripts/train.py --model vit_unet_4skip --dataset original --data_path ./dataset_orig --save_dir ./models_results_orig --tag vit_unet_4skip_orig --epochs 100 --early_stop_patience 15 --pos_weight 1.5 --seed 42
```

Run the three paper seeds for the proposed model:

```bash
bash scripts/run_all_seeds.sh refined
bash scripts/run_all_seeds.sh original
```

Evaluate one checkpoint:

```bash
python scripts/evaluate.py --model vit_unet --model_path models_results/vit_unet_best.pth
```

## Repository Contents

```text
scripts/              training, evaluation, metrics, and figure generation
models_results/       refined-dataset CSV results
models_results_orig/  original Deep-SAR CSV results
figures/              generated paper figures
requirements.txt      Python dependencies
```
