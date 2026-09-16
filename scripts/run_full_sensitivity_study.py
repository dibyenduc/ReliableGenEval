"""Run the small-sample sensitivity study across all four SummEval metrics
for model 5 vs model 9 (Cohen's d = -0.73 on coherence; near-null on
consistency). n_trials=500 for tighter coverage estimates than the
n_trials=200 exploratory runs.

Sample sizes capped at [25, 50, 100]: each model has exactly one scored
summary per article, and SummEval has 100 human-annotated articles, so
n=100 is the real full-sample ceiling, not an artificial ceiling we imposed.
"""

import os

import numpy as np
import pandas as pd

from evalcore.analysis.sensitivity import run_sensitivity_analysis, summarize_sensitivity
from evalcore.data.summeval import load_summeval

MODEL_A, MODEL_B = 5, 9
METRICS = ["coherence", "consistency", "fluency", "relevance"]
SAMPLE_SIZES = [25, 50, 100]
N_TRIALS = 500

os.makedirs("results", exist_ok=True)
df = load_summeval()

all_summaries = []
for metric in METRICS:
    a_full = df[df["model_index"] == MODEL_A][metric].reset_index(drop=True)
    b_full = df[df["model_index"] == MODEL_B][metric].reset_index(drop=True)
    doc_ids_full = df[df["model_index"] == MODEL_A]["doc_id"].reset_index(drop=True)

    full_estimate = float(np.asarray(a_full).mean() - np.asarray(b_full).mean())

    results_df = run_sensitivity_analysis(
        a_full, b_full, doc_ids_full,
        sample_sizes=SAMPLE_SIZES,
        n_trials=N_TRIALS,
        random_state=42,
    )
    summary_df = summarize_sensitivity(results_df, full_sample_estimate=full_estimate)
    summary_df.insert(0, "metric", metric)

    results_df.to_csv(f"results/sensitivity_{metric}_raw.csv", index=False)
    summary_df.to_csv(f"results/sensitivity_{metric}_summary.csv", index=False)
    all_summaries.append(summary_df)

    print(f"\n=== {metric} (full-sample estimate: {full_estimate:.4f}) ===")
    print(summary_df.drop(columns=["metric"]).to_string(index=False))

combined = pd.concat(all_summaries, ignore_index=True)
combined.to_csv("results/sensitivity_all_metrics_summary.csv", index=False)
print("\nSaved combined summary to results/sensitivity_all_metrics_summary.csv")

