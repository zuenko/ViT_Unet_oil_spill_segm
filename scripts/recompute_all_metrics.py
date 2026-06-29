"""
Single source of truth: recompute val/test metrics for trained checkpoints
on either the refined or the original dataset, plus aggregate mean +- std
from the existing 3-seed training histories.

Usage:
    python scripts/recompute_all_metrics.py --dataset refined
    python scripts/recompute_all_metrics.py --dataset original
    python scripts/recompute_all_metrics.py --ablation       # ViT-UNet variants only

Outputs (per --dataset):
    refined  -> models_results/recomputed_metrics.csv
                models_results/recomputed_per_sample.csv
                models_results/all_models_mean_std.csv          (overwritten)
    original -> models_results_orig/recomputed_metrics.csv
                models_results_orig/recomputed_per_sample.csv
                models_results_orig/all_models_mean_std.csv     (new)

Ablation mode:
    -> models_results/detailed_ablation.csv

Reuses:
    - models.{ViTUNet, UNetBaseline, ...}             from scripts/models.py
    - OilSpillDataset, get_transforms, load_dataset_orig from scripts/dataset.py
"""
import os, sys, time, glob, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    ViTUNet, UNetBaseline, DeepLabV3Plus, CBDNet,
    TransUNet, DSUNet, LRAUNet, DAENet, SegFormerWrapper,
    ViTUNetDeepSkip,
)
from dataset import OilSpillDataset, get_transforms, load_dataset_orig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main 9-model registry (display-name -> filename, builder, returns_tuple_for_boundary)
CKPT = {
    'ViT-UNet':   ('vit_unet_best.pth',     lambda: ViTUNet(model_name='vit_small_patch16_224', pretrained=False), False),
    'U-Net':      ('unet_best.pth',         lambda: UNetBaseline(),                                                False),
    'DeepLabV3+': ('deeplabv3plus_best.pth',lambda: DeepLabV3Plus(pretrained=False),                               False),
    'CBD-Net':    ('cbdnet_best.pth',       lambda: CBDNet(pretrained=False),                                      True),
    'TransUNet':  ('transunet_best.pth',    lambda: TransUNet(pretrained=False),                                   False),
    'DS-UNet':    ('dsunet_best.pth',       lambda: DSUNet(),                                                      True),
    'LRA-UNet':   ('lraunet_best.pth',      lambda: LRAUNet(),                                                     False),
    'DAENet':     ('daenet_best.pth',       lambda: DAENet(pretrained=False),                                      False),
    'SegFormer':  ('segformer_best.pth',    lambda: SegFormerWrapper(pretrained=False),                            False),
}

# Ablation registry: ViT-UNet variants whose best.pth lives under models_results/
ABLATION_CKPT = {
    'ViT-UNet (full)':       ('vit_unet_best.pth',           lambda: ViTUNet(model_name='vit_small_patch16_224', pretrained=False), False),
    'ViT-UNet (deep skip)':  ('vit_unet_deep_skip_best.pth', lambda: ViTUNetDeepSkip(pretrained=False),                              False),
}


def aggregate_metrics(tp, fp, tn, fn):
    eps = 1e-6
    iou      = tp / (tp + fp + fn + eps)
    iou_bg   = tn / (tn + fp + fn + eps)
    miou     = (iou + iou_bg) / 2
    prec     = tp / (tp + fp + eps)
    rec      = tp / (tp + fn + eps)
    f1       = 2 * prec * rec / (prec + rec + eps)
    acc      = (tp + tn) / (tp + tn + fp + fn + eps)
    spec     = tn / (tn + fp + eps)
    return dict(IoU=iou, mIoU=miou, F1=f1, Precision=prec, Recall=rec,
                Accuracy=acc, Specificity=spec)


@torch.no_grad()
def evaluate_model(model, loader, returns_tuple=False, store_per_sample=True, file_names=None):
    model.eval()
    tp = fp = tn = fn = 0
    per_sample = []

    sample_idx = 0
    for images, masks in tqdm(loader, leave=False):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        out = model(images)
        if returns_tuple:
            out = out[0]
        probs = torch.sigmoid(out)
        preds = (probs >= 0.5).float()

        if store_per_sample:
            preds_cpu = preds.cpu().numpy()
            masks_cpu = masks.cpu().numpy()
            for i in range(preds_cpu.shape[0]):
                p = preds_cpu[i].squeeze()
                m = masks_cpu[i].squeeze()
                _tp = float(((p == 1) & (m == 1)).sum())
                _fp = float(((p == 1) & (m == 0)).sum())
                _fn = float(((p == 0) & (m == 1)).sum())
                _tn = float(((p == 0) & (m == 0)).sum())
                tp += _tp; fp += _fp; fn += _fn; tn += _tn
                per_sample.append(dict(
                    sample=file_names[sample_idx] if file_names else sample_idx,
                    tp=_tp, fp=_fp, fn=_fn, tn=_tn,
                    iou=_tp/(_tp+_fp+_fn+1e-6),
                    recall=_tp/(_tp+_fn+1e-6),
                    precision=_tp/(_tp+_fp+1e-6),
                    gt_pixels=_tp+_fn,
                ))
                sample_idx += 1
        else:
            p = preds.flatten()
            m = masks.flatten()
            tp += float(((p == 1) & (m == 1)).sum())
            fp += float(((p == 1) & (m == 0)).sum())
            fn += float(((p == 0) & (m == 1)).sum())
            tn += float(((p == 0) & (m == 0)).sum())

    return aggregate_metrics(tp, fp, tn, fn), per_sample


def build_loader_refined(data_path, batch_size):
    """Find refined val/ under various layouts and return loader + file_names."""
    candidates = [
        (os.path.join(data_path, 'val', 'images'),                  os.path.join(data_path, 'val', 'masks')),
        (os.path.join(data_path, 'images', 'val'),                  os.path.join(data_path, 'masks', 'val')),
        (os.path.join(data_path, 'images', 'images', 'val'),        os.path.join(data_path, 'masks', 'masks', 'val')),
    ]
    for ai, am in candidates:
        if os.path.isdir(ai) and os.path.isdir(am):
            val_img_dir, val_mask_dir = ai, am
            break
    else:
        raise FileNotFoundError(f'refined val dir not found under {data_path}')
    print(f'[refined] images: {val_img_dir}')
    print(f'[refined] masks:  {val_mask_dir}')
    _, _, transform_val, transform_mask = get_transforms()
    ds = OilSpillDataset(val_img_dir, val_mask_dir, transform_val, transform_mask)
    file_names = list(ds.image_files)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), file_names, len(ds)


def build_loader_original(data_path, batch_size):
    """Use load_dataset_orig() on the test split (palsar + sentinel concat)."""
    _, _, transform_val, transform_mask = get_transforms()
    ds = load_dataset_orig(data_path, split='test', transform_img=transform_val, transform_mask=transform_mask)
    if ds is None:
        raise FileNotFoundError(f'original test dir not found under {data_path}')
    # ConcatDataset: build file_names by walking sub-datasets
    file_names = []
    for sub in ds.datasets:
        prefix = sub.sensor_type or 'unknown'
        for fn in sub.image_files:
            file_names.append(f'{prefix}/{fn}')
    print(f'[original] palsar+sentinel concat | total samples: {len(ds)}')
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), file_names, len(ds)


def run_evaluation(loader, file_names, ckpt_dir, registry):
    """Evaluate every checkpoint in `registry` against `loader`."""
    rows = []
    per_sample_rows = []
    for name, (ckpt_file, builder, ret_tuple) in registry.items():
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        if not os.path.isfile(ckpt_path):
            print(f'[SKIP] {name}: no checkpoint at {ckpt_path}')
            continue
        print(f'\n=== {name} ===')
        model = builder().to(DEVICE)
        sd = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(sd, strict=True)
        n_params_M = sum(p.numel() for p in model.parameters()) / 1e6

        t0 = time.time()
        metrics, per_sample = evaluate_model(model, loader, returns_tuple=ret_tuple,
                                             store_per_sample=True, file_names=file_names)
        elapsed = time.time() - t0

        row = dict(
            Model=name, Checkpoint=ckpt_file,
            Params_M=round(n_params_M, 3),
            Eval_seconds=round(elapsed, 2),
            **{k: round(v, 6) for k, v in metrics.items()},
        )
        print({k: f'{v:.4f}' if isinstance(v, float) else v for k, v in row.items()})
        rows.append(row)

        for s in per_sample:
            s['Model'] = name
            per_sample_rows.append(s)

        del model
        torch.cuda.empty_cache()
    return rows, per_sample_rows


def aggregate_seed_histories(history_dir, prefix=''):
    """
    Read {prefix}{Model}_seed{42,123,7}_history.csv files in `history_dir`,
    take the row with the LOWEST val_loss per seed (same model-selection
    criterion used by scripts/train.py to save *_best.pth) and compute
    mean +- std across the 3 seeds.

    Returns a DataFrame with the same column schema as the existing
    models_results/all_models_mean_std.csv:
        Model, mIoU_mean, mIoU_std, F1_mean, F1_std, ...
    """
    pattern = os.path.join(history_dir, f'{prefix}*_seed*_history.csv')
    files = glob.glob(pattern)
    by_model = {}
    for f in files:
        base = os.path.basename(f)
        # Strip prefix/_seedN_history.csv to extract model name
        name = base[len(prefix):] if prefix and base.startswith(prefix) else base
        # name is like "ViT-UNet_seed42_history.csv"
        if '_seed' not in name or '_history.csv' not in name:
            continue
        model_name = name.split('_seed')[0]
        by_model.setdefault(model_name, []).append(f)

    rows = []
    for model_name, paths in sorted(by_model.items()):
        bests = {'mIoU': [], 'F1': [], 'Precision': [], 'Recall': []}
        for p in paths:
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f'[WARN] cannot read {p}: {e}')
                continue
            if df.empty or 'mIoU' not in df.columns:
                continue
            # Selection criterion: lowest validation loss (matches train.py).
            # Fallback to highest mIoU only if the history CSV has no val_loss column.
            if 'val_loss' in df.columns:
                best_idx = df['val_loss'].idxmin()
            else:
                best_idx = df['mIoU'].idxmax()
            for k in bests:
                if k in df.columns:
                    bests[k].append(float(df.iloc[best_idx][k]))
        if not bests['mIoU']:
            continue
        rows.append({
            'Model':           model_name,
            'mIoU_mean':       float(np.mean(bests['mIoU'])),
            'mIoU_std':        float(np.std(bests['mIoU'])),
            'F1_mean':         float(np.mean(bests['F1'])),
            'F1_std':          float(np.std(bests['F1'])),
            'Precision_mean':  float(np.mean(bests['Precision'])),
            'Precision_std':   float(np.std(bests['Precision'])),
            'Recall_mean':     float(np.mean(bests['Recall'])),
            'Recall_std':      float(np.std(bests['Recall'])),
            'n_seeds':         len(bests['mIoU']),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['refined', 'original'], default='refined',
                        help='Which dataset variant to evaluate.')
    parser.add_argument('--data_path', default=None,
                        help='Override dataset root. Defaults: ./dataset (refined) or ./dataset_orig (original).')
    parser.add_argument('--out_dir', default=None,
                        help='Override output dir. Defaults: ./models_results (refined) or ./models_results_orig (original).')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ablation', action='store_true',
                        help='Evaluate ViT-UNet ablation variants only (refined dataset).')
    parser.add_argument('--no_aggregate', action='store_true',
                        help='Skip seed-history aggregation step.')
    args = parser.parse_args()

    # ---------- Ablation mode (always uses refined val) ----------
    if args.ablation:
        out_dir = args.out_dir or './models_results'
        data_path = args.data_path or './dataset'
        loader, file_names, n = build_loader_refined(data_path, args.batch_size)
        print(f'val samples: {n}')
        rows, _ = run_evaluation(loader, file_names, out_dir, ABLATION_CKPT)
        if not rows:
            print('[ERROR] no ablation checkpoints found.')
            return
        df = pd.DataFrame(rows).sort_values('mIoU', ascending=False).reset_index(drop=True)
        out_csv = os.path.join(out_dir, 'detailed_ablation.csv')
        df.to_csv(out_csv, index=False)
        print(f'\nSaved ablation -> {out_csv}')
        print(df.to_string(index=False))
        return

    # ---------- Main flow: refined or original ----------
    if args.dataset == 'refined':
        data_path = args.data_path or './dataset'
        out_dir   = args.out_dir   or './models_results'
        loader, file_names, n = build_loader_refined(data_path, args.batch_size)
        history_prefix = ''
    else:
        data_path = args.data_path or './dataset_orig'
        out_dir   = args.out_dir   or './models_results_orig'
        loader, file_names, n = build_loader_original(data_path, args.batch_size)
        history_prefix = 'orig_'

    print(f'samples: {n}')
    os.makedirs(out_dir, exist_ok=True)

    rows, per_sample_rows = run_evaluation(loader, file_names, out_dir, CKPT)

    df = pd.DataFrame(rows).sort_values('mIoU', ascending=False).reset_index(drop=True)
    out_csv = os.path.join(out_dir, 'recomputed_metrics.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nSaved aggregate -> {out_csv}')
    print(df.to_string(index=False))

    df_ps = pd.DataFrame(per_sample_rows)
    out_ps = os.path.join(out_dir, 'recomputed_per_sample.csv')
    df_ps.to_csv(out_ps, index=False)
    print(f'Saved per-sample -> {out_ps} ({len(df_ps)} rows)')

    if not args.no_aggregate:
        print(f'\n=== Aggregating 3-seed histories from {out_dir} (prefix="{history_prefix}") ===')
        df_seed = aggregate_seed_histories(out_dir, prefix=history_prefix)
        if not df_seed.empty:
            out_seed = os.path.join(out_dir, 'all_models_mean_std.csv')
            df_seed = df_seed.sort_values('mIoU_mean', ascending=False).reset_index(drop=True)
            df_seed.to_csv(out_seed, index=False)
            print(f'Saved mean+-std -> {out_seed}')
            print(df_seed.to_string(index=False))
        else:
            print('[WARN] no seed-history files found; skipping aggregation.')


if __name__ == '__main__':
    main()
