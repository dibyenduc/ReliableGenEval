"""Summarize the position-bias study: per-dimension inconsistency rates,
and a pass/escalate flag for using the pairwise judge on each dimension."""

import pandas as pd

from evalcore.judge.bias import POSITION_BIAS_THRESHOLD, summarize_position_bias

INPUT_PATH = "results/position_bias_study.csv"
OUTPUT_PATH = "results/position_bias_summary.csv"

df = pd.read_csv(INPUT_PATH)
summary = summarize_position_bias(df)
summary.to_csv(OUTPUT_PATH, index=False)

print(f"Position-bias study: {df['doc_id'].nunique()} pairs x {df['dimension'].nunique()} dimensions "
      f"= {len(df)} comparisons\n")
print(f"Overall inconsistency rate: {(~df['consistent']).mean():.1%}\n")

for _, row in summary.iterrows():
    verdict = "ESCALATE (position-unreliable)" if row["escalate"] else "USABLE"
    print(f"  {row['dimension']}: {verdict}")
    print(f"    inconsistency={row['inconsistency_rate']:.1%} "
          f"({row['n_inconsistent']}/{row['n_comparisons']})")
    print(f"    reason: {row['reason']}\n")

