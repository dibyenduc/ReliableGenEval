"""Scale demonstration: run pointwise judge calls across a large sample of
real SummEval data through MockCloudBackend at high concurrency, proving
out checkpointing/resume and bounded concurrency at a volume far beyond
the earlier Ollama-based studies (which ran at concurrency=1, ~15 items).

This intentionally uses MockCloudBackend (zero cost, no API key needed)
to demonstrate that swapping OllamaBackend for a real rate-limited cloud
backend is the ONLY change needed to go from proof-of-concept to
production scale -- everything else (job construction, telemetry,
checkpointing, dashboard) is backend-agnostic.

Re-running this script a second time against the same TELEMETRY_PATH
should complete near-instantly with 0 jobs run, proving checkpoint/resume
survives a real process exit, not just an in-memory test double.
"""

import time

from evalcore.data.summeval import load_summeval
from evalcore.judge.backends import MockCloudBackend
from evalcore.judge.pointwise import build_pointwise_prompt, parse_pointwise_score
from evalcore.orchestration.runner import JudgeJob, run_jobs

N_ITEMS = 500  # ~30x the largest single-backend study run so far
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
TELEMETRY_PATH = "results/scale_demo_telemetry.jsonl"
MODEL = "mock-cloud-large-v1"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} hours"


def main() -> None:
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

    print(f"Total jobs: {len(jobs)} ({len(sample)} items x {len(DIMENSIONS)} dimensions)")

    backend = MockCloudBackend(
        max_concurrency=200,
        mean_latency_s=0.3,
        latency_jitter_s=0.15,
        failure_rate=0.03,
        seed=7,
    )

    start = time.time()
    results = run_jobs(jobs, backend, MODEL, TELEMETRY_PATH, max_concurrency=200, max_retries=3)
    elapsed = time.time() - start

    if not results:
        print("\nAll jobs already completed in a prior run (checkpoint/resume skipped everything).")
        print(f"Elapsed: {elapsed:.2f}s")
        return

    n_parse_ok = sum(1 for r in results if r.event.parse_ok)
    n_errors = sum(1 for r in results if r.event.error is not None)
    n_retried = sum(1 for r in results if r.event.retry_count > 0)
    rate = len(results) / elapsed

    print(f"\nCompleted {len(results)} jobs in {elapsed:.1f}s ({rate:.1f} jobs/sec)")
    print(f"Parse OK: {n_parse_ok}/{len(results)} ({n_parse_ok / len(results):.1%})")
    print(f"Required at least one retry: {n_retried} ({n_retried / len(results):.1%})")
    print(f"Failed even after retries: {n_errors} ({n_errors / len(results):.1%})")
    print(f"\nTelemetry written to {TELEMETRY_PATH}")

    full_dataset_jobs = len(df) * len(DIMENSIONS)
    print(f"\nAt {rate:.0f} jobs/sec, projected time for:")
    for label, n_jobs in [
        ("full SummEval dataset", full_dataset_jobs),
        ("10x SummEval (multi-dataset)", full_dataset_jobs * 10),
        ("1M-item production workload", 1_000_000 * len(DIMENSIONS)),
    ]:
        print(f"  {label} ({n_jobs:,} jobs): {format_duration(n_jobs / rate)}")


if __name__ == "__main__":
    main()

