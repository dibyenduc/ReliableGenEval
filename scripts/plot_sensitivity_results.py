"""Plot sensitivity study results: CI width and empirical coverage vs
sample size, faceted by metric, comparing two model pairs to show whether
findings (e.g. consistency undercoverage) replicate across pairs."""

import glob

import matplotlib.pyplot as plt
import pandas as pd

files = glob.glob("results/sensitivity_all_metrics_*_summary.csv")
if not files:
    raise FileNotFoundError("No combined summary CSVs found in results/. Run run_full_sensitivity_study.py first.")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
metrics = sorted(df["metric"].unique())
pairs = sorted(df["model_pair"].unique())

fig, axes = plt.subplots(2, len(metrics), figsize=(4 * len(metrics), 8), sharex=True)

for col, metric in enumerate(metrics):
    sub = df[df["metric"] == metric]
    ax_width = axes[0, col]
    ax_cov = axes[1, col]
    for pair in pairs:
        pair_sub = sub[sub["model_pair"] == pair].sort_values("sample_size")
        ax_width.plot(pair_sub["sample_size"], pair_sub["mean_ci_width"], marker="o", label=pair)
        ax_cov.errorbar(
            pair_sub["sample_size"], pair_sub["empirical_coverage"],
            yerr=1.96 * pair_sub["coverage_se"], marker="o", capsize=3, label=pair,
        )
    ax_width.set_title(metric)
    ax_width.set_ylabel("Mean CI width" if col == 0 else "")
    ax_cov.axhline(0.95, color="gray", linestyle="--", linewidth=1, label="nominal 95%")
    ax_cov.set_ylabel("Empirical coverage" if col == 0 else "")
    ax_cov.set_xlabel("Sample size (n)")
    ax_cov.set_ylim(0.5, 1.05)

axes[0, 0].legend(fontsize=8)
axes[1, 0].legend(fontsize=8)
fig.suptitle("Bootstrap CI width and coverage vs sample size, by metric and model pair", fontsize=13)
fig.tight_layout()
fig.savefig("results/sensitivity_plot.png", dpi=150)
print("Saved plot to results/sensitivity_plot.png")

