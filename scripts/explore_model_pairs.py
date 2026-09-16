"""One-off exploration: rank SummEval model pairs by effect size on
coherence, to choose a deliberate (not arbitrary) pair for the Week 2
small-sample sensitivity study."""

import itertools

import numpy as np

from evalcore.data.summeval import load_summeval

df = load_summeval()

results = []
for i, j in itertools.combinations(range(16), 2):
    a = df[df["model_index"] == i]["coherence"].values
    b = df[df["model_index"] == j]["coherence"].values
    mean_diff = a.mean() - b.mean()
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    results.append((i, j, mean_diff, cohens_d))

results.sort(key=lambda r: abs(r[3]), reverse=True)

print("Top 5 largest effect sizes (by |Cohen's d| on coherence):")
for i, j, md, d in results[:5]:
    print(f"  model {i} vs model {j}: mean_diff={md:+.3f}, cohens_d={d:+.3f}")

print("\nMedian effect size pairs (around the middle of the distribution):")
mid = len(results) // 2
for i, j, md, d in results[mid - 2 : mid + 3]:
    print(f"  model {i} vs model {j}: mean_diff={md:+.3f}, cohens_d={d:+.3f}")

print(f"\nTotal pairs: {len(results)}")
print(f"Effect size range: {results[-1][3]:.3f} to {results[0][3]:.3f}")
