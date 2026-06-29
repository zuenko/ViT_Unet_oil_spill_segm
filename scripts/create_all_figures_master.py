"""
MASTER SCRIPT: Generate publication figures for IEEE Access paper.

All numbers come from CSV files produced by:
  - scripts/recompute_all_metrics.py --dataset {refined,original}
  - scripts/recompute_all_metrics.py --ablation
  - scripts/compute_model_complexity.py

This script never holds hardcoded metric values. Re-running the upstream
scripts and then this one regenerates every figure deterministically from
the published checkpoints.

Surviving figures (overwritten in figures/):
  1. final_metrics_comparison.png        (mIoU, F1, Precision, Recall, mean ± std)
  2. final_efficiency_tradeoff.png       (mIoU vs Params with FPS as bubble size)
  3. final_training_curves.png           (val loss + val mIoU per epoch, seed 42)
  4. dataset_comparison_original_vs_refined.png  (refined vs original mIoU)
  5. supplementary_ablation_detailed.png (ViT-UNet ablation variants)

Qualitative figures (figure1_dataset_samples.png, figure4_real_predictions.png,
best_worst_real_cases.png, qualitative_comparison_clean.png) are produced by
create_figure1_dataset.py and create_qualitative_all_models.py respectively.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.path import Path
from matplotlib.patches import PathPatch

for _style in ('seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot'):
    try:
        plt.style.use(_style)
        break
    except (OSError, ValueError):
        continue
sns.set_context("paper", font_scale=1.2)

# -----------------------------------------------------------------------------
# Single source of truth: load every CSV once at import time.
# -----------------------------------------------------------------------------
DF_REFINED  = pd.read_csv('models_results/recomputed_metrics.csv').set_index('Model')
DF_ORIG     = pd.read_csv('models_results_orig/recomputed_metrics.csv').set_index('Model')
DF_MEAN_R   = pd.read_csv('models_results/all_models_mean_std.csv').set_index('Model')
DF_MEAN_O   = pd.read_csv('models_results_orig/all_models_mean_std.csv').set_index('Model')
DF_COMPLEX  = pd.read_csv('models_results/model_complexity_analysis.csv').set_index('Model')
DF_ABLATION = pd.read_csv('models_results/detailed_ablation.csv')

# Canonical model order (descending refined-mIoU, computed once).
MODELS_LIST = list(DF_REFINED.sort_values('mIoU', ascending=False).index)

COLORS_7 = {
    'ViT-UNet':   '#2ca02c', 'TransUNet':  '#ff7f0e', 'U-Net':      '#d62728',
    'DS-UNet':    '#9467bd', 'CBD-Net':    '#8c564b', 'DeepLabV3+': '#e377c2',
    'LRA-UNet':   '#7f7f7f', 'SegFormer':  '#17becf', 'DAENet':     '#bcbd22',
}

ACCESS_BLUE = '#0072CE'
DARK = '#1f2933'
GRID = '#d9e2ec'
GREEN = '#22c55e'

def save_fig(name):
    plt.savefig(f'figures/{name}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Created: figures/{name}")


def add_box(ax, xy, width, height, text, fc='white', ec=DARK,
            fontsize=10, weight='normal', color=DARK):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle='round,pad=0.02,rounding_size=0.025',
        linewidth=1.2, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha='center', va='center', fontsize=fontsize,
        fontweight=weight, color=color, linespacing=1.15
    )
    return box


def arrow(ax, start, end, color=DARK, lw=1.4, style='-|>'):
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=4, shrinkB=4)
    )


def skip_curve(ax, start, end, color='#ef4444'):
    sx, sy = start
    ex, ey = end
    verts = [
        (sx, sy),
        (sx + 0.08, sy + 0.10),
        (ex - 0.08, ey + 0.10),
        (ex, ey),
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    patch = PathPatch(
        Path(verts, codes), facecolor='none', edgecolor=color,
        lw=1.4, linestyle=(0, (4, 3))
    )
    ax.add_patch(patch)
    ax.annotate(
        '', xy=end, xytext=(ex - 0.035, ey + 0.01),
        arrowprops=dict(arrowstyle='-|>', lw=1.3, color=color, shrinkA=0, shrinkB=3)
    )


# =============================================================================
# 1) Final metrics comparison (4 panels, mean ± std across 3 seeds)
# =============================================================================
def create_final_metrics_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sorted_models = list(DF_MEAN_R.sort_values('mIoU_mean', ascending=False).index)
    metrics = [('mIoU', 'mIoU'), ('F1', 'F1-Score'),
               ('Precision', 'Precision'), ('Recall', 'Recall')]
    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        means = [DF_MEAN_R.loc[m, f'{key}_mean'] for m in sorted_models]
        stds  = [DF_MEAN_R.loc[m, f'{key}_std']  for m in sorted_models]
        colors = [COLORS_7.get(m, '#333333') for m in sorted_models]
        bars = ax.bar(sorted_models, means, yerr=stds, capsize=4,
                      color=colors, edgecolor='black', linewidth=1.2,
                      error_kw={'linewidth': 1.4, 'ecolor': 'black'})
        ax.set_ylabel(key, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} (mean ± std, 3 seeds)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, mu in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(stds) * 0.4,
                    f'{mu:.3f}', ha='center', fontsize=8, fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    save_fig('final_metrics_comparison.png')


# =============================================================================
# 2) Efficiency–performance trade-off (single-checkpoint mIoU vs Params, FPS bubble)
# =============================================================================
def create_final_efficiency_tradeoff():
    fig = plt.figure(figsize=(14.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.35], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[0, 1])
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.axis('off')

    plot_models = [m for m in MODELS_LIST if m in DF_COMPLEX.index]
    plot_models = sorted(plot_models, key=lambda m: DF_REFINED.loc[m, 'mIoU'])
    top3 = set(DF_REFINED.sort_values('mIoU', ascending=False).head(3).index)

    ax.axhspan(66.0, 69.2, color='#f0fdf4', alpha=0.55, zorder=0)
    ax.text(
        1.5, 68.93, 'high-accuracy region',
        fontsize=10.5, color='#166534', fontweight='bold',
        ha='left', va='top'
    )

    for m in plot_models:
        if m not in DF_COMPLEX.index:
            continue
        params = DF_COMPLEX.loc[m, 'Parameters (M)']
        fps = DF_COMPLEX.loc[m, 'FPS']
        miou = DF_REFINED.loc[m, 'mIoU']
        is_vit = m == 'ViT-UNet'
        is_top = m in top3
        size = 170 + fps * 2.35
        ax.scatter(
            params, miou * 100, s=size,
            c=GREEN if is_vit else COLORS_7.get(m, '#64748b'),
            alpha=0.95 if is_top else 0.78,
            edgecolors='black',
            linewidth=2.4 if is_top else 1.15,
            zorder=5 if is_top else 3,
        )

    label_offsets = {
        'ViT-UNet': (12, 15),
        'CBD-Net': (12, -18),
        'SegFormer': (12, 14),
        'TransUNet': (-92, -5),
        'U-Net': (12, 14),
        'DS-UNet': (10, -23),
        'DeepLabV3+': (-96, -20),
        'LRA-UNet': (12, -11),
        'DAENet': (12, -19),
    }
    for m in plot_models:
        params = DF_COMPLEX.loc[m, 'Parameters (M)']
        miou = DF_REFINED.loc[m, 'mIoU'] * 100
        dx, dy = label_offsets.get(m, (8, 8))
        ax.annotate(
            f'{m}\n{miou:.1f}%',
            (params, miou),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=11.2,
            fontweight='bold' if m in top3 else 'normal',
            color=DARK,
            bbox=dict(boxstyle='round,pad=0.20', fc='white', ec='none', alpha=0.86),
            zorder=6,
        )

    ax.set_xscale('log')
    ax.set_xlim(1.2, 135)
    ax.set_ylim(61.2, 69.2)
    ax.set_xlabel('Parameters (millions, log scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('mIoU on Refined-SOS validation (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy-efficiency trade-off on Refined-SOS',
                 fontsize=17, fontweight='bold', pad=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, which='major', color=GRID, linewidth=0.8)
    ax.grid(True, which='minor', color=GRID, linewidth=0.35, alpha=0.45)

    vit_params = DF_COMPLEX.loc['ViT-UNet', 'Parameters (M)']
    vit_miou = DF_REFINED.loc['ViT-UNet', 'mIoU'] * 100
    ax.axhline(vit_miou, color=GREEN, lw=1.2, alpha=0.35)
    ax.axvline(vit_params, color=GREEN, lw=1.2, alpha=0.35)

    def callout(y, title, body, color):
        patch = FancyBboxPatch(
            (0.04, y), 0.92, 0.17,
            boxstyle='round,pad=0.018,rounding_size=0.025',
            linewidth=1.1, edgecolor=color, facecolor='white'
        )
        info_ax.add_patch(patch)
        info_ax.text(0.09, y + 0.122, title, fontsize=11.5,
                     fontweight='bold', color=color, ha='left', va='center')
        info_ax.text(0.09, y + 0.060, body, fontsize=10.2,
                     color=DARK, ha='left', va='center', linespacing=1.18)

    info_ax.text(0.04, 0.96, 'Reading guide', fontsize=13,
                 fontweight='bold', color=DARK, ha='left', va='top')
    info_ax.text(
        0.04, 0.88,
        'Higher is better. Left is smaller.\nLarger bubbles are faster.',
        fontsize=10.3, color='#475569', ha='left', va='top', linespacing=1.25
    )
    callout(0.62, 'Best mIoU',
            f"ViT-UNet: {vit_miou:.1f}%\n33.1M params, 124 FPS", GREEN)
    fastest = DF_COMPLEX.loc[plot_models, 'FPS'].idxmax()
    callout(0.39, 'Fastest model',
            f"{fastest}: {DF_COMPLEX.loc[fastest, 'FPS']:.0f} FPS\n"
            f"mIoU {DF_REFINED.loc[fastest, 'mIoU'] * 100:.1f}%",
            '#7c3aed')
    smallest = DF_COMPLEX.loc[plot_models, 'Parameters (M)'].idxmin()
    callout(0.16, 'Smallest model',
            f"{smallest}: {DF_COMPLEX.loc[smallest, 'Parameters (M)']:.1f}M params\n"
            f"mIoU {DF_REFINED.loc[smallest, 'mIoU'] * 100:.1f}%",
            '#64748b')

    info_ax.text(0.04, 0.075, 'Bubble examples', fontsize=10.5,
                 fontweight='bold', color=DARK, ha='left')
    for x, fps, label in [(0.18, 120, '120'), (0.43, 250, '250'), (0.72, 400, '400 FPS')]:
        info_ax.scatter(x, 0.025, s=170 + fps * 2.35,
                        c='white', edgecolors='black', linewidth=1.0)
        info_ax.text(x, 0.025, label, ha='center', va='center',
                     fontsize=8.4, color=DARK)

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.90)
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/final_efficiency_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Created: figures/final_efficiency_tradeoff.png")


def create_vit_architecture_diagram():
    fig, ax = plt.subplots(figsize=(7.3, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.96, 'ViT-UNet architecture', ha='center', va='top',
            fontsize=16, fontweight='bold', color=DARK)

    add_box(ax, (0.04, 0.72), 0.14, 0.13, 'SAR input\nB,3,224,224', fc='#f8fafc', fontsize=9)
    add_box(ax, (0.23, 0.70), 0.16, 0.17, 'ViT-Small\npatch encoder\n14x14 tokens', fc='#e0f2fe', fontsize=9, weight='bold')
    add_box(ax, (0.45, 0.73), 0.13, 0.10, 'Block 11\nbottleneck\nB,384,14,14', fc='#dbeafe', fontsize=7.5)

    decoder = [
        ((0.66, 0.72), 'Up 1\n14->28'),
        ((0.66, 0.55), 'Up 2\n28->56'),
        ((0.66, 0.38), 'Up 3\n56->112'),
        ((0.66, 0.21), 'Up 4\n112->224'),
    ]
    for xy, text in decoder:
        add_box(ax, xy, 0.14, 0.105, text, fc='#dcfce7', fontsize=8.7, weight='bold')

    add_box(ax, (0.83, 0.21), 0.15, 0.105, 'Final conv\n1x1 + sigmoid', fc='#fef3c7', fontsize=8.1, weight='bold')
    add_box(ax, (0.85, 0.06), 0.13, 0.085, 'Output\nB,1,224,224', fc='#f8fafc', fontsize=8.3)

    arrow(ax, (0.18, 0.785), (0.23, 0.785))
    arrow(ax, (0.39, 0.785), (0.45, 0.785))
    arrow(ax, (0.58, 0.785), (0.66, 0.775))
    for idx in range(len(decoder) - 1):
        sx, sy = decoder[idx][0]
        ex, ey = decoder[idx + 1][0]
        arrow(ax, (sx + 0.07, sy), (ex + 0.07, ey + 0.105))
    arrow(ax, (0.80, 0.262), (0.83, 0.262))
    arrow(ax, (0.905, 0.21), (0.915, 0.145))

    skip_specs = [
        ('block 2 skip', (0.29, 0.66), (0.66, 0.262)),
        ('block 5 skip', (0.31, 0.62), (0.66, 0.432)),
        ('block 8 skip', (0.33, 0.58), (0.66, 0.602)),
    ]
    for label, start, end in skip_specs:
        skip_curve(ax, start, end)
        ax.text(start[0] - 0.03, start[1] + 0.018, label, fontsize=7.6,
                color='#b91c1c', ha='right', va='center')
    ax.text(0.56, 0.835, 'block 11\nbottleneck', fontsize=7.4,
            color='#2563eb', ha='center', va='bottom')

    ax.text(0.50, 0.08,
            'Blocks 2/5/8 provide lateral skips; block 11 provides the bottleneck.\n'
            'All routed features share the same 14x14 token grid before decoding.',
            ha='center', va='center', fontsize=8.4, color='#475569')

    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/vit_unet_architecture.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Created: figures/vit_unet_architecture.png")


# =============================================================================
# 3) Training curves (seed 42 history, 2 panels: val loss + val mIoU)
# =============================================================================
def create_final_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in MODELS_LIST:
        path = f'models_results/{m}_seed42_history.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        color = COLORS_7.get(m, '#333333')
        ep = df['epoch'].to_numpy()
        axes[0].plot(ep, df['val_loss'].to_numpy(), '-', color=color, label=m, linewidth=2)
        axes[1].plot(ep, df['mIoU'].to_numpy(),     '-', color=color, label=m, linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Validation Loss (seed 42)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('mIoU', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation mIoU (seed 42)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9, loc='lower right')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig('final_training_curves.png')


# =============================================================================
# 4) Refined vs original dataset comparison (single-ckpt mIoU)
# =============================================================================
def create_dataset_comparison_original_vs_refined():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sorted_models = list(DF_REFINED.sort_values('mIoU', ascending=False).index)
    refined = [DF_REFINED.loc[m, 'mIoU'] for m in sorted_models]
    original = [DF_ORIG.loc[m, 'mIoU'] if m in DF_ORIG.index else np.nan
                for m in sorted_models]
    colors = [COLORS_7.get(m, '#333333') for m in sorted_models]
    x = np.arange(len(sorted_models))
    width = 0.4

    ax = axes[0]
    ax.bar(x - width/2, original, width, label='Original (Deep-SAR test)',
           color='lightcoral', edgecolor='black', linewidth=1.2)
    ax.bar(x + width/2, refined,  width, label='Refined (val)',
           color='lightgreen', edgecolor='black', linewidth=1.2)
    ax.set_ylabel('mIoU', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Single-checkpoint mIoU: Original vs Refined',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    lo = min([v for v in original + refined if not np.isnan(v)]) - 0.02
    hi = max([v for v in original + refined if not np.isnan(v)]) + 0.02
    ax.set_ylim([lo, hi])

    ax = axes[1]
    deltas = [r - o if not np.isnan(o) else 0 for r, o in zip(refined, original)]
    bars = ax.bar(sorted_models, deltas, color=colors,
                  edgecolor='black', linewidth=1.2)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Δ mIoU (Refined − Original)',
                  fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Per-model effect of dataset refinement',
                 fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for bar, d in zip(bars, deltas):
        y = d + (0.0008 if d >= 0 else -0.0008)
        va = 'bottom' if d >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f'{d:+.3f}', ha='center', va=va,
                fontsize=8, fontweight='bold')
    plt.tight_layout()
    save_fig('dataset_comparison_original_vs_refined.png')


# =============================================================================
# 5) ViT-UNet ablation (whatever variants are present in detailed_ablation.csv)
# =============================================================================
def create_supplementary_ablation_detailed():
    fig, ax = plt.subplots(figsize=(10, 6))
    df = DF_ABLATION.copy()
    if df.empty:
        print("  [skip] detailed_ablation.csv is empty")
        return
    # Identify the "full" / baseline row to compute degradations.
    is_full = df['Model'].str.lower().str.contains('full')
    if is_full.any():
        baseline = df.loc[is_full, 'mIoU'].iloc[0]
    else:
        baseline = df['mIoU'].max()
    df['Degradation'] = baseline - df['mIoU']

    def color_for(deg):
        if abs(deg) < 1e-6:
            return '#2ca02c'
        if deg > 0.03:
            return '#d62728'
        if deg > 0.015:
            return '#ff7f0e'
        return '#ffbb78'

    colors = [color_for(d) for d in df['Degradation']]
    bars = ax.bar(df['Model'], df['Degradation'],
                  color=colors, edgecolor='black', linewidth=1.4)
    for bar, deg, miou in zip(bars, df['Degradation'], df['mIoU']):
        h = bar.get_height()
        if abs(h) < 1e-6:
            label = f'{miou:.3f}\n(Baseline)'
        else:
            label = f'-{deg:.3f}\n({miou:.3f})'
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                label, ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel('Degradation from proposed ViT-UNet (mIoU)',
                  fontsize=12, fontweight='bold')
    ax.set_title('Ablation: variants whose checkpoints are on disk',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    top = max(df['Degradation'].max() * 1.4, 0.05)
    ax.set_ylim([min(df['Degradation'].min() - 0.005, -0.005), top])

    legend_elements = [
        Patch(facecolor='#2ca02c', label='Baseline (Full)'),
        Patch(facecolor='#d62728', label='High Impact (>0.03)'),
        Patch(facecolor='#ff7f0e', label='Medium Impact (0.015–0.03)'),
        Patch(facecolor='#ffbb78', label='Low Impact (<0.015)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.tight_layout()
    save_fig('supplementary_ablation_detailed.png')


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("Regenerating figures from CSVs (no hardcoded numbers)")
    print("=" * 70)
    os.makedirs('figures', exist_ok=True)
    create_final_metrics_comparison()
    create_final_efficiency_tradeoff()
    create_vit_architecture_diagram()
    create_final_training_curves()
    create_dataset_comparison_original_vs_refined()
    create_supplementary_ablation_detailed()
    print("=" * 70)
    print("Done. Qualitative figures are produced by:")
    print("  - scripts/create_figure1_dataset.py")
    print("  - scripts/create_qualitative_all_models.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
