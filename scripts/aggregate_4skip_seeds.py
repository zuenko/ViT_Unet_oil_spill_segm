"""
Aggregate ViT-UNet 4-skip results across 3 seeds.

Reads vit_unet_4skip_seed{42,123,7}_best.pth, evaluates each on the refined
val set with the same protocol as the other models, and writes:
  - models_results/vit_unet_4skip_per_seed.csv (one row per seed)
  - models_results/vit_unet_4skip_summary.csv (mean +/- std across 3 seeds)
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vit_unet_plus import ViTUNet4Skip
from dataset import OilSpillDataset, get_transforms, load_dataset_orig
from utils import MetricsCalculator, set_seed


def evaluate_ckpt(ckpt_path, val_loader, device):
    model = ViTUNet4Skip(model_name="vit_small_patch16_224", pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    mc = MetricsCalculator()
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            preds = (torch.sigmoid(out) > 0.5).float()
            mc.update(preds, masks)
    return mc.get_metrics()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default='./dataset')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7])
    p.add_argument('--save_dir', type=str, default='./models_results')
    p.add_argument('--tag', type=str, default='vit_unet_4skip')
    p.add_argument('--dataset', type=str, default='refined',
                   choices=['refined', 'original'],
                   help='refined: --data_path ./dataset; '
                        'original: --data_path ./dataset_orig (eval on test split)')
    args = p.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, transform_val, transform_mask = get_transforms()
    if args.dataset == 'original':
        val_ds = load_dataset_orig(args.data_path, split='test',
                                   transform_img=transform_val,
                                   transform_mask=transform_mask)
        if val_ds is None:
            raise FileNotFoundError(
                f"--dataset original expects palsar+sentinel under {args.data_path}/test")
    else:
        val_img = os.path.join(args.data_path, 'val', 'images')
        val_msk = os.path.join(args.data_path, 'val', 'masks')
        if not os.path.exists(val_img):
            # nested structure used by training script
            if os.path.exists(os.path.join(args.data_path, 'images', 'images', 'val')):
                val_img = os.path.join(args.data_path, 'images', 'images', 'val')
                val_msk = os.path.join(args.data_path, 'masks', 'masks', 'val')
            else:
                val_img = os.path.join(args.data_path, 'images', 'val')
                val_msk = os.path.join(args.data_path, 'masks', 'val')
        val_ds = OilSpillDataset(val_img, val_msk, transform_val, transform_mask)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_ds)}")

    rows = []
    for s in args.seeds:
        ckpt = os.path.join(args.save_dir, f"{args.tag}_seed{s}_best.pth")
        if not os.path.exists(ckpt):
            print(f"WARN: {ckpt} missing, skipping")
            continue
        print(f"Evaluating seed {s} ...")
        m = evaluate_ckpt(ckpt, val_loader, device)
        row = {'seed': s, 'mIoU': m['mIoU'], 'F1': m['F1'],
               'Precision': m['Precision'], 'Recall': m['Recall'],
               'Specificity': m['Specificity'], 'Accuracy': m['Accuracy']}
        rows.append(row)
        print(f"  mIoU={m['mIoU']:.4f} F1={m['F1']:.4f} P={m['Precision']:.4f} R={m['Recall']:.4f}")

    if not rows:
        print("No checkpoints found.")
        return

    df = pd.DataFrame(rows)
    per_seed_path = os.path.join(args.save_dir, f"{args.tag}_per_seed.csv")
    df.to_csv(per_seed_path, index=False)
    print(f"\nWrote: {per_seed_path}")
    print(df.to_string(index=False))

    metrics = ['mIoU', 'F1', 'Precision', 'Recall', 'Specificity', 'Accuracy']
    summary = {'model': args.tag, 'n_seeds': len(df)}
    for m in metrics:
        summary[f'{m}_mean'] = df[m].mean()
        summary[f'{m}_std'] = df[m].std() if len(df) > 1 else 0.0
    sdf = pd.DataFrame([summary])
    sum_path = os.path.join(args.save_dir, f"{args.tag}_summary.csv")
    sdf.to_csv(sum_path, index=False)
    print(f"\nWrote: {sum_path}")
    print(f"\n=== SUMMARY (mean ± std over {len(df)} seeds) ===")
    for m in metrics:
        print(f"  {m:<12} {summary[f'{m}_mean']:.4f} ± {summary[f'{m}_std']:.4f}")


if __name__ == '__main__':
    main()
