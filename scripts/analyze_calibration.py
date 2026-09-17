"""Analyze the judge-vs-human calibration study results: correlation,
error, and scale-usage comparison between judge and human scores."""

import pandas as pd

from evalcore.analysis.calibration import compute_calibration_by_dimension

df = pd.read_csv("results/judge_calibration_raw.csv")
df = df[df["parse_ok"]].copy()
df["judge_score"] = df["judge_score"].astype(float)

print(f"Total scored pairs: {len(df)} (parse failures excluded: {(~pd.read_csv('results/judge_calibration_raw.csv')['parse_ok']).sum()})")

print("\n=== Score distribution: judge vs human (rounded) ===")
print("Judge score value counts:")
print(df["judge_score"].value_counts().sort_index())
print("\nHuman score value counts (rounded to nearest int):")
print(df["human_score"].round().astype(int).value_counts().sort_index())

calibration = compute_calibration_by_dimension(df)
calibration.to_csv("results/judge_calibration_metrics.csv", index=False)
print("\n=== Calibration metrics by dimension ===")
print(calibration.to_string(index=False))
