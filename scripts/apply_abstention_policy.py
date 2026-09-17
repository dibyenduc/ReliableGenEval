"""Build the abstention policy from the real calibration study results
and show which dimensions would be escalated to human review."""

import pandas as pd

from evalcore.judge.abstention import build_abstention_policy

calibration_df = pd.read_csv("results/judge_calibration_metrics.csv")
policy = build_abstention_policy(calibration_df)

print("Abstention policy built from real calibration data:\n")
for dimension, reliability in policy.dimension_reliability.items():
    status = "TRUSTED" if reliability.trusted else "ESCALATE TO HUMAN"
    print(f"  {dimension}: {status}")
    print(f"    r={reliability.pearson_r:.3f}, p={reliability.pearson_p:.4g}, n={reliability.n}")
    print(f"    reason: {reliability.reason}\n")

raw_df = pd.read_csv("results/judge_calibration_raw.csv")
annotated = policy.annotate(raw_df)
annotated.to_csv("results/judge_calibration_with_abstention.csv", index=False)
print(f"Escalation rate: {annotated['escalate_to_human'].mean():.1%} of all judge calls")

