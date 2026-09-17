"""Generate the observability dashboard from the real scale-demo telemetry
log (2,000 events, MockCloudBackend, 200-way concurrency)."""

from evalcore.orchestration.dashboard import generate_dashboard

output_path = generate_dashboard(
    telemetry_path="results/scale_demo_telemetry.jsonl",
    output_path="results/scale_demo_dashboard.html",
    title="ReliableGenEval -- Scale Demonstration (2,000 jobs, MockCloudBackend)",
)

print(f"Dashboard written to {output_path}")
