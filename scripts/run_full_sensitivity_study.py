"""Run the small-sample sensitivity study across all four SummEval metrics
for a given model pair. n_trials=500 for tighter coverage estimates.

Sample sizes capped at [25, 50, 100]: each model has exactly one scored
summary per article, and SummEval has 100 human-annotated articles, so
n=100 is the real full-sample ceiling, not an artificial ceiling we imposed.

Usage:
    uv run python scripts/run_full_sensitivity_study.py --model-a 5 --model-b 9
"""

import argparse
import os

import numpy as np
import pandas as pd

from evalcore.analysis.sensitivity import run_sensitivity_analysis, summarize_sensitivity
from evalcore.data.summeval import load_summeval

METRICS = ["coherence", "consistency", "fluency", "relevance"]
SAMPLE_SIZES = [25, 50, 100]
N_TRIALS = 500


def main(model_a: int, model_b: int) -> None:
    os.makedirs("results", exist_ok=True)
    df = load_summeval()
    pair_tag = f"m{model_a}v{model_b}"

    all_summaries = []
    for metric in METRICS:
        a_full = df[df["model_index"] == model_a][metric].reset_index(drop=True)
        b_full = df[df["model_index"] == model_b][metric].reset_index(drop=True)
        doc_ids_full = df[df["model_index"] == model_a]["doc_id"].reset_index(drop=True)

        full_estimate = float(np.asarray(a_full).mean() - np.asarray(b_full).mean())

        results_df = run_sensitivity_analysis(
            a_full, b_full, doc_ids_full,
            sample_sizes=SAMPLE_SIZES,
            n_trials=N_TRIALS,
            random_state=42,
        )
        summary_df = summarize_sensitivity(results_df, full_sample_estimate=full_estimate)
        summary_df.insert(0, "metric", metric)
        summary_df.insert(0, "model_pair", pair_tag)

        results_df.to_csv(f"results/sensitivity_{metric}_{pair_tag}_raw.csv", index=False)
        summary_df.to_csv(f"results/sensitivity_{metric}_{pair_tag}_summary.csv", index=False)
        all_summaries.append(summary_df)

        print(f"\n=== {metric} | {pair_tag} (full-sample estimate: {full_estimate:.4f}) ===")
        print(summary_df.drop(columns=["metric", "model_pair"]).to_string(index=False))

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(f"results/sensitivity_all_metrics_{pair_tag}_summary.csv", index=False)
    print(f"\nSaved combined summary to results/sensitivity_all_metrics_{pair_tag}_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=int, required=True)
    parser.add_argument("--model-b", type=int, required=True)
    args = parser.parse_args()
    main(args.model_a, args.model_b)

