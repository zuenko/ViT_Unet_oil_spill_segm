"""
Look-alike strata + per-sensor stratification analysis for the Refined-SOS val
split.

Two outputs the paper needs:

A) Per-sensor stratification (PALSAR L-band vs Sentinel-1 C-band): grouped
   pixel-aggregated mIoU/F1/Precision/Recall for every model. Sensor is
   inferred from the val image filename prefix (palsar_*.png / sentinel_*.png),
   so no external metadata is required.

B) Texture-based look-alike clustering of the val *background* regions
   (gt==0 pixels): for each val image we extract a low-dimensional texture
   descriptor (intensity moments, GLCM Haralick features, gradient stats,
   LBP entropy), standardise, and K-means into K=4 clusters. We then report
   per-cluster mean F1 (with bootstrap 95% CI) for every model. The cluster
   labels are post-hoc named by inspecting cluster centroids and example
   images (see figures/lookalike_clusters.png).

Inputs:
  - dataset/images/images/val/*.png            (1615 val images, palsar_/sentinel_ prefix)
  - dataset/masks/masks/val/*.png              (matching binary masks)
  - models_results/recomputed_per_sample.csv   (per-image TP/FP/FN/TN for 9 models;
                                               the "ViT-UNet" row here is the legacy
                                               3-skip v1, so we ALSO run inference
                                               for vit_unet_4skip_seed42 and add it
                                               as "ViT-UNet (4-skip)").

Outputs:
  models_results/per_sensor_metrics.csv
  models_results/lookalike_clusters.csv          (sample, cluster_id, cluster_label)
  models_results/lookalike_strata_fpr.csv        (Model, cluster, n, F1_mean, F1_lo, F1_hi,
                                                  Precision, Recall, FPR_pixel)
  models_results/lookalike_cluster_centroids.csv (feature-space centroids in standardised units)
  figures/lookalike_clusters.png                 (centroid bar chart + example tiles per cluster)
"""
from __future__ import annotations

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# A) Per-sensor stratification
# --------------------------------------------------------------------------
def per_sensor_table(per_sample_csv: str, out_csv: str) -> pd.DataFrame:
    df = pd.read_csv(per_sample_csv)
    # `sample` is either "palsar_0.png" (refined) or "palsar/0.png" (orig);
    # extract sensor as the first token before "/" or "_"
    def sensor_of(s: str) -> str:
        s = str(s)
        if '/' in s:
            return s.split('/')[0]
        return s.split('_')[0]
    df['sensor'] = df['sample'].map(sensor_of)
    eps = 1e-6
    rows = []
    for (model, sensor), g in df.groupby(['Model', 'sensor']):
        tp, fp, fn, tn = g[['tp', 'fp', 'fn', 'tn']].sum()
        iou_oil = tp / (tp + fp + fn + eps)
        iou_bg = tn / (tn + fp + fn + eps)
        miou = (iou_oil + iou_bg) / 2
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        fpr = fp / (fp + tn + eps)
        rows.append(dict(Model=model, sensor=sensor, n_images=len(g),
                         mIoU=miou, F1=f1, Precision=prec, Recall=rec,
                         FPR_pixel=fpr))
    out = pd.DataFrame(rows).sort_values(['Model', 'sensor'])
    out.to_csv(out_csv, index=False)
    print(f'Saved per-sensor -> {out_csv} ({len(out)} rows)')
    return out


# --------------------------------------------------------------------------
# B) Texture feature extraction on background pixels
# --------------------------------------------------------------------------
FEATURE_NAMES = [
    'bg_mean', 'bg_std', 'bg_skew', 'bg_kurt',
    'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
    'grad_mean', 'grad_std', 'lbp_entropy',
]


def _to_gray_u8(img_array: np.ndarray) -> np.ndarray:
    if img_array.ndim == 3:
        img = img_array.mean(axis=-1)
    else:
        img = img_array
    return img.astype(np.uint8)


def extract_features(img_array: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    img = _to_gray_u8(img_array)
    bg_pixels = img[gt_mask == 0]
    if bg_pixels.size < 100:
        bg_pixels = img.flatten()
    bg_pixels = bg_pixels.astype(np.float32)

    bg_mean = float(bg_pixels.mean())
    bg_std = float(bg_pixels.std()) + 1e-6
    bg_skew = float(stats.skew(bg_pixels)) if bg_std > 1e-6 else 0.0
    bg_kurt = float(stats.kurtosis(bg_pixels)) if bg_std > 1e-6 else 0.0

    # GLCM on full image, 64-level quantisation, 1-px distance, 4 orientations
    img_q = (img >> 2).astype(np.uint8)  # 256 -> 64 levels
    glcm = graycomatrix(
        img_q,
        distances=[1],
        angles=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64, symmetric=True, normed=True,
    )
    glcm_contrast = float(graycoprops(glcm, 'contrast').mean())
    glcm_homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    glcm_energy = float(graycoprops(glcm, 'energy').mean())
    glcm_correlation = float(graycoprops(glcm, 'correlation').mean())

    # gradient magnitude stats on background
    gy, gx = np.gradient(img.astype(np.float32))
    gmag = np.sqrt(gx * gx + gy * gy)
    bg_grad = gmag[gt_mask == 0] if (gt_mask == 0).any() else gmag.flatten()
    grad_mean = float(bg_grad.mean())
    grad_std = float(bg_grad.std())

    # LBP entropy (uniform, 8 neighbours, radius 1) over background
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_bg = lbp[gt_mask == 0] if (gt_mask == 0).any() else lbp.flatten()
    hist, _ = np.histogram(lbp_bg, bins=10, range=(0, 10), density=True)
    lbp_entropy = float(-(hist * np.log(hist + 1e-12)).sum())

    return np.array([
        bg_mean, bg_std, bg_skew, bg_kurt,
        glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation,
        grad_mean, grad_std, lbp_entropy,
    ], dtype=np.float32)


def build_feature_table(img_dir: str, mask_dir: str) -> pd.DataFrame:
    img_files = sorted(os.listdir(img_dir))
    rows = []
    for fname in tqdm(img_files, desc='texture features'):
        ipath = os.path.join(img_dir, fname)
        mpath = os.path.join(mask_dir, fname)
        if not os.path.isfile(mpath):
            continue
        img = np.array(Image.open(ipath).convert('L'))
        mask = (np.array(Image.open(mpath).convert('L')) > 127).astype(np.uint8)
        feats = extract_features(img, mask)
        rows.append([fname] + list(feats) + [float(mask.mean())])
    df = pd.DataFrame(rows, columns=['sample'] + FEATURE_NAMES + ['oil_frac'])
    return df


# --------------------------------------------------------------------------
# K-means clustering and semantic labelling of cluster centroids
# --------------------------------------------------------------------------
def cluster_and_label(feat_df: pd.DataFrame, K: int = 3, random_state: int = 42):
    """
    Cluster within each sensor separately (K clusters per sensor). This removes
    the sensor-fingerprint axis from the texture space, so each cluster captures
    phenomenological sea-state / backscatter regimes rather than sensor type.
    Final cluster ids are 0..2K-1; sensor is encoded in cluster_label.
    """
    feat_df = feat_df.copy()
    feat_df['sensor'] = feat_df['sample'].astype(str).map(
        lambda s: (s.split('/')[0] if '/' in s else s.split('_')[0])
    )

    cluster_ids = np.full(len(feat_df), -1, dtype=int)
    centroids_all = []  # one row per cluster in standardised units (per-sensor)
    labels = {}
    cid_offset = 0
    for sensor in sorted(feat_df['sensor'].unique()):
        mask = feat_df['sensor'].values == sensor
        X = feat_df.loc[mask, FEATURE_NAMES].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=K, n_init=20, random_state=random_state).fit(Xs)
        local_labels = _name_clusters(km.cluster_centers_, sensor=sensor)
        for k in range(K):
            cid_global = cid_offset + k
            cluster_ids[np.where(mask)[0][km.labels_ == k]] = cid_global
            labels[cid_global] = f'{sensor.upper()}: {local_labels[k]}'
            centroids_all.append(km.cluster_centers_[k])
        cid_offset += K

    feat_df['cluster_id'] = cluster_ids
    feat_df['cluster_label'] = feat_df['cluster_id'].map(labels)
    return feat_df, np.array(centroids_all), labels


def _name_clusters(centroids: np.ndarray, sensor: str | None = None) -> dict:
    """
    Assign a short phenomenological label per cluster based on the dominant
    standardised feature directions of its centroid (within-sensor z-units).

    Ranking is purely on sea-state proxies:
      - low bg_std + low gradient + high homogeneity      ~ "Calm sea / low-wind"
      - high bg_std + high gradient + high contrast        ~ "Rough sea / high-wind"
      - intermediate                                       ~ "Moderate sea"
    """
    K = centroids.shape[0]
    idx = {n: FEATURE_NAMES.index(n) for n in FEATURE_NAMES}

    # Single "roughness" score; smallest=calmest, largest=roughest
    rough = np.array([
        c[idx['bg_std']] + c[idx['grad_mean']] + c[idx['glcm_contrast']]
        - c[idx['glcm_homogeneity']]
        for c in centroids
    ])
    order = np.argsort(rough)
    names_by_rank = ['Calm sea / low-wind',
                     'Moderate sea',
                     'Rough sea / high-wind']
    # if more than 3 clusters, pad linearly
    while len(names_by_rank) < K:
        names_by_rank.insert(len(names_by_rank) // 2, 'Moderate sea')
    out = {}
    for rank, k in enumerate(order):
        out[int(k)] = names_by_rank[rank]
    return out


# --------------------------------------------------------------------------
# Per-model per-cluster metrics with bootstrap 95% CI on image-level F1
# --------------------------------------------------------------------------
def per_cluster_metrics(per_sample_df: pd.DataFrame,
                        cluster_df: pd.DataFrame,
                        n_boot: int = 2000,
                        random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    merged = per_sample_df.merge(cluster_df[['sample', 'cluster_id', 'cluster_label']],
                                 on='sample', how='inner')
    # image-level F1
    eps = 1e-6
    merged['f1_img'] = 2 * merged['precision'] * merged['recall'] / (
        merged['precision'] + merged['recall'] + eps)
    merged['fpr_img'] = merged['fp'] / (merged['fp'] + merged['tn'] + eps)

    rows = []
    for (model, cid, clab), g in merged.groupby(['Model', 'cluster_id', 'cluster_label']):
        f1s = g['f1_img'].values
        if len(f1s) == 0:
            continue
        f1_mean = float(f1s.mean())
        # bootstrap CI on the mean F1
        idx = rng.integers(0, len(f1s), size=(n_boot, len(f1s)))
        boots = f1s[idx].mean(axis=1)
        f1_lo, f1_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

        tp, fp, fn, tn = g[['tp', 'fp', 'fn', 'tn']].sum()
        prec_pix = tp / (tp + fp + eps)
        rec_pix = tp / (tp + fn + eps)
        f1_pix = 2 * prec_pix * rec_pix / (prec_pix + rec_pix + eps)
        fpr_pix = fp / (fp + tn + eps)
        rows.append(dict(
            Model=model, cluster_id=int(cid), cluster_label=clab,
            n_images=int(len(g)),
            F1_mean=f1_mean, F1_lo=f1_lo, F1_hi=f1_hi,
            F1_pixel=f1_pix, Precision_pixel=prec_pix, Recall_pixel=rec_pix,
            FPR_pixel=fpr_pix,
        ))
    return pd.DataFrame(rows).sort_values(['Model', 'cluster_id'])


# --------------------------------------------------------------------------
# Inference for ViT-UNet 4-skip seed-42 to add a clean per-sample row
# under model name "ViT-UNet (4-skip)" — paper's main "ViT-UNet (ours)".
# --------------------------------------------------------------------------
@torch.no_grad()
def eval_4skip_per_sample(ckpt_path: str, data_path: str) -> pd.DataFrame:
    from vit_unet_plus import ViTUNet4Skip
    from dataset import OilSpillDataset, get_transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, transform_val, transform_mask = get_transforms()
    candidates = [
        (os.path.join(data_path, 'val', 'images'), os.path.join(data_path, 'val', 'masks')),
        (os.path.join(data_path, 'images', 'val'), os.path.join(data_path, 'masks', 'val')),
        (os.path.join(data_path, 'images', 'images', 'val'),
         os.path.join(data_path, 'masks', 'masks', 'val')),
    ]
    for ai, am in candidates:
        if os.path.isdir(ai) and os.path.isdir(am):
            val_img_dir, val_mask_dir = ai, am
            break
    else:
        raise FileNotFoundError(f'refined val dir not found under {data_path}')

    ds = OilSpillDataset(val_img_dir, val_mask_dir, transform_val, transform_mask)
    file_names = list(ds.image_files)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    model = ViTUNet4Skip(model_name='vit_small_patch16_224', pretrained=False).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    rows = []
    sample_idx = 0
    eps = 1e-6
    for images, masks in tqdm(loader, desc='ViT-UNet (4-skip) inference', leave=False):
        images, masks = images.to(device), masks.to(device)
        out = model(images)
        preds = (torch.sigmoid(out) >= 0.5).float()
        p_np = preds.cpu().numpy()
        m_np = masks.cpu().numpy()
        for i in range(p_np.shape[0]):
            p = p_np[i].squeeze()
            m = m_np[i].squeeze()
            tp = float(((p == 1) & (m == 1)).sum())
            fp = float(((p == 1) & (m == 0)).sum())
            fn = float(((p == 0) & (m == 1)).sum())
            tn = float(((p == 0) & (m == 0)).sum())
            rows.append(dict(
                sample=file_names[sample_idx],
                tp=tp, fp=fp, fn=fn, tn=tn,
                iou=tp / (tp + fp + fn + eps),
                recall=tp / (tp + fn + eps),
                precision=tp / (tp + fp + eps),
                gt_pixels=tp + fn,
                Model='ViT-UNet (4-skip)',
            ))
            sample_idx += 1
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------
# Visualisation: centroid bar chart + example tiles per cluster
# --------------------------------------------------------------------------
def make_cluster_figure(feat_df: pd.DataFrame,
                        centroids_std: np.ndarray,
                        labels: dict,
                        img_dir: str,
                        out_png: str,
                        n_examples: int = 1,
                        random_state: int = 42):
    """Compact two-panel figure designed to remain readable at LaTeX
    \columnwidth. Top: K x F z-score heatmap of cluster centroids. Bottom:
    one example tile per cluster (sample closest to the cluster centroid)."""
    cids = sorted(labels.keys())
    K = len(cids)
    F = len(FEATURE_NAMES)

    fig = plt.figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(2, K, height_ratios=[1.0, 1.4], hspace=0.45, wspace=0.18)

    # ---- Top panel: heatmap of standardised centroids (K x F) ----
    ax_h = fig.add_subplot(gs[0, :])
    vmax = float(np.max(np.abs(centroids_std)))
    im = ax_h.imshow(centroids_std, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                     aspect='auto')
    ax_h.set_xticks(np.arange(F))
    ax_h.set_xticklabels(FEATURE_NAMES, rotation=35, ha='right', fontsize=8)
    ax_h.set_yticks(np.arange(K))
    yticklabels = [f'C{k}: {labels[k]}  (n={int((feat_df.cluster_id==k).sum())})'
                   for k in cids]
    ax_h.set_yticklabels(yticklabels, fontsize=8)
    for i in range(K):
        for j in range(F):
            v = centroids_std[i, j]
            ax_h.text(j, i, f'{v:+.1f}', ha='center', va='center',
                      fontsize=6.5,
                      color='white' if abs(v) > 0.55 * vmax else 'black')
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.025, pad=0.01)
    cbar.set_label('within-sensor z-score', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_h.set_title('Centroid feature signature  (rows: clusters; columns: 11-dim texture descriptor)',
                   fontsize=10)

    # ---- Bottom panel: representative tile per cluster (closest to centroid)
    rng = np.random.default_rng(random_state)
    feature_cols = ['feat_' + name for name in FEATURE_NAMES] if 'feat_bg_mean' in feat_df.columns else FEATURE_NAMES
    have_feats = all(col in feat_df.columns for col in feature_cols)
    for col, k in enumerate(cids):
        ax_img = fig.add_subplot(gs[1, col])
        members = feat_df[feat_df['cluster_id'] == k]
        if len(members) == 0:
            ax_img.axis('off')
            continue
        if have_feats:
            X = members[feature_cols].values
            mu = X.mean(axis=0)
            sd = X.std(axis=0) + 1e-9
            Xz = (X - mu) / sd
            target = centroids_std[col]
            d = np.linalg.norm(Xz - target, axis=1)
            fn = members.iloc[int(np.argmin(d))]['sample']
        else:
            fn = rng.choice(members['sample'].values)
        img = np.array(Image.open(os.path.join(img_dir, fn)).convert('L'))
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(f'C{k}', fontsize=9)
        ax_img.axis('off')

    fig.suptitle('Within-sensor texture clusters of the Refined-SOS validation backgrounds '
                 '(K=3 per sensor)', fontsize=11, y=0.995)
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'Saved figure -> {out_png}')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./dataset')
    parser.add_argument('--per_sample_csv', default='./models_results/recomputed_per_sample.csv')
    parser.add_argument('--out_dir', default='./models_results')
    parser.add_argument('--fig_path', default='./figures/lookalike_clusters.png')
    parser.add_argument('--vit_4skip_ckpt', default='./models_results/vit_unet_4skip_seed42_best.pth')
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.fig_path) or '.', exist_ok=True)

    # ------------ Locate val image / mask dirs ------------
    candidates = [
        (os.path.join(args.data_path, 'val', 'images'),
         os.path.join(args.data_path, 'val', 'masks')),
        (os.path.join(args.data_path, 'images', 'val'),
         os.path.join(args.data_path, 'masks', 'val')),
        (os.path.join(args.data_path, 'images', 'images', 'val'),
         os.path.join(args.data_path, 'masks', 'masks', 'val')),
    ]
    for ai, am in candidates:
        if os.path.isdir(ai) and os.path.isdir(am):
            img_dir, mask_dir = ai, am
            break
    else:
        raise FileNotFoundError('refined val dir not found')
    print(f'val images: {img_dir}')
    print(f'val masks:  {mask_dir}')

    # ------------ A) per-sensor table (existing 9 models) ------------
    print('\n[A] Per-sensor stratification ...')
    df_persensor_9 = per_sensor_table(args.per_sample_csv,
                                      os.path.join(args.out_dir, 'per_sensor_metrics.csv'))

    # ------------ Inference for 4-skip ViT-UNet (seed 42) ------------
    if os.path.isfile(args.vit_4skip_ckpt):
        print(f'\n[*] Evaluating ViT-UNet (4-skip) from {args.vit_4skip_ckpt}')
        df_4skip = eval_4skip_per_sample(args.vit_4skip_ckpt, args.data_path)
        df_4skip.to_csv(os.path.join(args.out_dir, 'vit_unet_4skip_per_sample.csv'), index=False)
        print(f'Saved 4-skip per-sample -> models_results/vit_unet_4skip_per_sample.csv '
              f'({len(df_4skip)} rows)')
    else:
        print(f'[WARN] {args.vit_4skip_ckpt} missing; 4-skip row will be omitted.')
        df_4skip = pd.DataFrame()

    # Combined per-sample dataframe for downstream cluster analysis
    df_ps_legacy = pd.read_csv(args.per_sample_csv)
    df_ps_all = (pd.concat([df_ps_legacy, df_4skip], ignore_index=True)
                 if not df_4skip.empty else df_ps_legacy)

    # Regenerate per-sensor table incorporating the 4-skip row
    tmp_path = os.path.join(args.out_dir, '_per_sample_combined.tmp.csv')
    df_ps_all.to_csv(tmp_path, index=False)
    per_sensor_table(tmp_path, os.path.join(args.out_dir, 'per_sensor_metrics.csv'))
    os.remove(tmp_path)

    # ------------ B) texture features + K-means clustering ------------
    print('\n[B] Extracting texture features ...')
    feat_csv = os.path.join(args.out_dir, 'val_texture_features.csv')
    if os.path.isfile(feat_csv):
        print(f'  (cached) reading {feat_csv}')
        feat_df = pd.read_csv(feat_csv)
    else:
        feat_df = build_feature_table(img_dir, mask_dir)
        feat_df.to_csv(feat_csv, index=False)
        print(f'Saved features -> {feat_csv} ({len(feat_df)} rows)')

    print(f'\n[B] K-means clustering (K={args.K}) ...')
    feat_df, centroids_std, labels = cluster_and_label(feat_df, K=args.K,
                                                       random_state=args.seed)
    cluster_csv = os.path.join(args.out_dir, 'lookalike_clusters.csv')
    feat_df[['sample', 'cluster_id', 'cluster_label']].to_csv(cluster_csv, index=False)
    print(f'Saved cluster assignments -> {cluster_csv}')

    cent_csv = os.path.join(args.out_dir, 'lookalike_cluster_centroids.csv')
    cent_df = pd.DataFrame(centroids_std, columns=FEATURE_NAMES)
    cent_df['cluster_id'] = np.arange(len(cent_df))
    cent_df['cluster_label'] = cent_df['cluster_id'].map(labels)
    cent_df['n_images'] = [
        int((feat_df['cluster_id'] == k).sum()) for k in range(len(cent_df))
    ]
    cent_df.to_csv(cent_csv, index=False)
    print(f'Saved centroids -> {cent_csv}')

    # ------------ Per-model per-cluster metrics ------------
    print('\n[B] Per-model per-cluster metrics (bootstrap CI) ...')
    strata = per_cluster_metrics(df_ps_all, feat_df, random_state=args.seed)
    strata_csv = os.path.join(args.out_dir, 'lookalike_strata_metrics.csv')
    strata.to_csv(strata_csv, index=False)
    print(f'Saved strata -> {strata_csv}')

    # ------------ Figure ------------
    print('\n[*] Building cluster figure ...')
    make_cluster_figure(feat_df, centroids_std, labels, img_dir,
                        args.fig_path, random_state=args.seed)

    # ------------ Console summary ------------
    print('\n=== CLUSTER SIZES ===')
    print(cent_df[['cluster_id', 'cluster_label', 'n_images']].to_string(index=False))

    print('\n=== ViT-UNet (4-skip) per-cluster F1 ===')
    main_model = 'ViT-UNet (4-skip)' if 'ViT-UNet (4-skip)' in strata['Model'].unique() else 'ViT-UNet'
    sub = strata[strata['Model'] == main_model]
    if not sub.empty:
        print(sub[['cluster_id', 'cluster_label', 'n_images',
                   'F1_mean', 'F1_lo', 'F1_hi', 'FPR_pixel']].to_string(index=False))

    print('\n=== PER-SENSOR mIoU (selected) ===')
    ps = pd.read_csv(os.path.join(args.out_dir, 'per_sensor_metrics.csv'))
    print(ps.pivot_table(index='Model', columns='sensor', values='mIoU').round(4).to_string())


if __name__ == '__main__':
    main()
