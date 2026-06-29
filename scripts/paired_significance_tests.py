"""Paired significance tests for ViT-UNet vs baselines on per-image confusion counts.

Reads existing per-sample CSVs (no model evaluation, no retraining), computes:
  - Paired bootstrap 95% CI on mean per-image IoU and Recall deltas (n_boot=10000)
  - McNemar test on per-image "improved vs degraded" sign counts
  - Wilcoxon signed-rank test on per-image IoU deltas

Outputs models_results/paired_significance_tests.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X<=k) under Binomial(n, 0.5), times 2 for two-sided
    p = 2.0 * stats.binom.cdf(k, n, 0.5)
    return min(1.0, p)


def paired_bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = deltas[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    base = pd.read_csv("models_results/recomputed_per_sample.csv")
    vit4 = pd.read_csv("models_results/vit_unet_4skip_per_sample.csv")

    # Reference = ViT-UNet (4-skip) on the refined val split.
    # Compare against every other model in recomputed_per_sample.csv.
    ref_name = vit4["Model"].iloc[0]  # "ViT-UNet (4-skip)"
    ref = vit4[["sample", "tp", "fp", "fn", "tn", "iou", "recall"]].copy()
    ref = ref.rename(columns={c: f"{c}_ref" for c in ["tp", "fp", "fn", "tn", "iou", "recall"]})

    rows = []
    for model in sorted(base["Model"].unique()):
        sub = base[base["Model"] == model][["sample", "tp", "fp", "fn", "tn", "iou", "recall"]].copy()
        merged = ref.merge(sub, on="sample", how="inner")
        if len(merged) == 0:
            continue

        d_iou = (merged["iou_ref"] - merged["iou"]).to_numpy()
        d_rec = (merged["recall_ref"] - merged["recall"]).to_numpy()

        # McNemar on per-image "improved vs degraded" w.r.t. IoU
        b = int((d_iou > 0).sum())  # ref better
        c = int((d_iou < 0).sum())  # baseline better
        p_mcnemar = mcnemar_exact(b, c)

        # Wilcoxon signed-rank on IoU deltas
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(d_iou, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, p_wilcoxon = float("nan"), float("nan")

        ci_iou_lo, ci_iou_hi = paired_bootstrap_ci(d_iou)
        ci_rec_lo, ci_rec_hi = paired_bootstrap_ci(d_rec)

        rows.append({
            "Reference": ref_name,
            "Baseline": model,
            "n_images": int(len(merged)),
            "mean_dIoU": float(d_iou.mean()),
            "dIoU_CI95_lo": ci_iou_lo,
            "dIoU_CI95_hi": ci_iou_hi,
            "mean_dRecall": float(d_rec.mean()),
            "dRecall_CI95_lo": ci_rec_lo,
            "dRecall_CI95_hi": ci_rec_hi,
            "n_ref_better": b,
            "n_baseline_better": c,
            "p_mcnemar": p_mcnemar,
            "p_wilcoxon": float(p_wilcoxon) if p_wilcoxon == p_wilcoxon else float("nan"),
        })

    df = pd.DataFrame(rows).sort_values("Baseline").reset_index(drop=True)
    out = "models_results/paired_significance_tests.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Wrote {out} ({len(df)} rows)")
    # Pretty console preview
    show = df[[
        "Baseline", "n_images", "mean_dIoU", "dIoU_CI95_lo", "dIoU_CI95_hi",
        "mean_dRecall", "p_mcnemar", "p_wilcoxon",
    ]].copy()
    for c in ["mean_dIoU", "dIoU_CI95_lo", "dIoU_CI95_hi", "mean_dRecall"]:
        show[c] = show[c].map(lambda x: f"{x:+.4f}")
    for c in ["p_mcnemar", "p_wilcoxon"]:
        show[c] = show[c].map(lambda x: f"{x:.2e}" if x < 1e-3 else f"{x:.4f}")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
