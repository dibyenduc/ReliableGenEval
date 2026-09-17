"""Scale demo, now wired with REAL calibration-driven escalation logic
(not a placeholder). Rebuilds the AbstentionPolicy from the actual Week 3
calibration results, then runs the same 500-item x 4-dimension pointwise
workload through MockCloudBackend, with escalate_fn reflecting genuine
per-dimension trust decisions.
"""

import time

import pandas as pd

from evalcore.analysis.calibration import compute_calibration_by_dimension
from evalcore.data.summeval import load_summeval
from evalcore.judge.abstention import build_abstention_policy
from evalcore.judge.backends import MockCloudBackend
from evalcore.judge.pointwise import build_pointwise_prompt, parse_pointwise_score
from evalcore.orchestration.escalation import make_pointwise_escalate_fn
from evalcore.orchestration.runner import JudgeJob, run_jobs

N_ITEMS = 500
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
TELEMETRY_PATH = "results/scale_demo_escalation_telemetry.jsonl"
MODEL = "mock-cloud-large-v1"

calibration_source = pd.read_csv("results/judge_calibration_with_abstention.csv")
calibration_df = compute_calibration_by_dimension(calibration_source)
policy = build_abstention_policy(calibration_df)

print("Rebuilt abstention policy from real calibration data:")
for dim, rel in policy.dimension_reliability.items():
    print(f"  {dim}: {'TRUSTED' if rel.trusted else 'ESCALATE'} (r={rel.pearson_r:.3f}, p={rel.pearson_p:.4g})")

escalate_fn = make_pointwise_escalate_fn(policy)

df = load_summeval()
sample = df.sample(n=min(N_ITEMS, len(df)), random_state=42)

jobs = [
    JudgeJob(
        item_id=str(row["doc_id"]) + f"::{row['model_index']}",
        dimension=dimension,
        judge_type="pointwise",
        prompt=build_pointwise_prompt(row["article"], row["summary"], dimension),
        parser=parse_pointwise_score,
    )
    for _, row in sample.iterrows()
    for dimension in DIMENSIONS
]

backend = MockCloudBackend(max_concurrency=200, mean_latency_s=0.3, latency_jitter_s=0.15, failure_rate=0.03, seed=7)

start = time.time()
results = run_jobs(jobs, backend, MODEL, TELEMETRY_PATH, max_concurrency=200, max_retries=3, escalate_fn=escalate_fn)
elapsed = time.time() - start

if results:
    n_escalated = sum(1 for r in results if r.event.escalated)
    print(f"\nCompleted {len(results)} jobs in {elapsed:.1f}s")
    print(f"Escalated to human review: {n_escalated}/{len(results)} ({n_escalated/len(results):.1%})")
    for dim in DIMENSIONS:
        dim_results = [r for r in results if r.dimension == dim]
        dim_escalated = sum(1 for r in dim_results if r.event.escalated)
        print(f"  {dim}: {dim_escalated}/{len(dim_results)} escalated ({dim_escalated/len(dim_results):.1%})")
else:
    print("\nAll jobs already completed in a prior run.")

