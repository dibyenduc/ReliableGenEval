# ReliableGenEval

**A statistically calibrated framework for knowing when LLM-as-judge pipelines can be trusted — and when they can't.**

LLM-as-judge is now standard practice for evaluating generative AI outputs at scale, but an LLM judge is itself an unvalidated model making subjective calls. This project empirically calibrates a judge against 100 human-annotated summaries and finds it's statistically trustworthy on 3 of 4 evaluation dimensions (r=0.48-0.55, p<3e-9) but not the 4th — then encodes that finding into an automatic escalation policy that routes untrustworthy dimensions to human review rather than trusting the model blindly.

It goes further to stress-test the judge under adversarial conditions: proving that a well-known LLM-judge failure mode (position bias in head-to-head comparisons) doesn't just exist but is total — 100% of disagreements deterministically favor whichever option is shown first — and that the standard fix (order-averaging via majority vote) mechanically fails to recover any real signal, a diagnostic result rather than a workaround. Two independent mitigation attempts (order-averaging, chain-of-thought prompting) each measurably improve but do not fully resolve this, a finding reported precisely rather than overstated.

Finally, the project proves this evaluation pipeline is scale-ready by architecture: a pluggable backend interface, verified bounded concurrency, checkpoint/resume validated against real process failures, and structured observability — demonstrated at 2,000-job proof-of-concept volume with a measured, extrapolated path to a 4-million-job production workload at $0 cost.

## Key Results

- **3 of 4 dimensions** statistically validated (coherence, consistency, relevance); fluency correctly flagged unreliable and auto-escalated
- **25% automatic escalation rate**, derived from calibration data and reproduced exactly at 2,000-job orchestration scale
- **66.7% → 41.7%** reduction in pairwise position-bias inconsistency via chain-of-thought prompting — a real, partial fix
- **520-684 jobs/sec** sustained throughput; checkpoint/resume verified in 0.01s against real process restarts
- **~2.1 hours** projected runtime for a simulated 4-million-job production workload

---

## Part 1: Judge Reliability Research

### Calibration Against Ground Truth

Built a **pointwise judge**: given one article and one summary, score the summary 1-5 on a single quality dimension (coherence, consistency, fluency, or relevance). Calibrated against SummEval — 100 real summaries with human-annotated ground-truth scores across all four dimensions.

| Dimension | Pearson r vs. human scores | p-value | Verdict |
|---|---|---|---|
| Coherence | 0.550 | 2.98 × 10⁻⁹ | Statistically validated |
| Consistency | 0.551 | 2.93 × 10⁻⁹ | Statistically validated |
| Relevance | 0.483 | 3.69 × 10⁻⁷ | Statistically validated |
| Fluency | 0.155 | 0.124 | Not statistically significant |

### Turning Calibration Into an Operational Policy

Built an **abstention policy** derived directly from the calibration statistics: a dimension is escalated to mandatory human review if its judge-human correlation is either not statistically significant, or too weak to be practically useful even when significant. Applied to real data: **25% of judge calls (fluency) are automatically escalated; 75% (coherence, consistency, relevance) are trusted** — a concrete, auditable rule rather than a subjective call.

### Stress-Testing: Position Bias in Pairwise Comparison

Built a **pairwise judge** (given two summaries, pick the better one: A/B/Tie) and tested for position bias by running each comparison in both presentation orders. Across 60 real comparisons (15 pairs × 4 dimensions):

- **66.7% of comparisons flipped their preferred summary** purely from swapping presentation order.
- In **100% of the 40 flipped cases**, the judge preferred whichever summary was shown first — a deterministic heuristic, not noise.
- Worst dimension: consistency (86.7% inconsistent) — the dimension most dependent on careful reading.

### Attempting the Standard Fix — and Discovering Why It Fails

Implemented order-averaging (run both orders; on disagreement, add a third randomized-order call and majority-vote). After correcting an initial tie-break bug that mechanically replayed original slot order, the corrected result was more revealing than reassuring: **44 of 44 tie-break verdicts (100%) matched whichever order the random coin flip selected**, independent of actual content.

This proves the model's pairwise verdict, in disagreement cases, carries **zero recoverable content signal** — a pure function of slot position. Majority-vote debiasing cannot fix this because there is no real signal to average toward.

### A Second Mitigation Attempt: Chain-of-Thought Prompting

Redesigned the pairwise prompt to require evidence-based reasoning before a structured verdict. Result: overall inconsistency dropped from 66.7% to **41.7%** — a real, substantial improvement. Consistency improved most (86.7% → 40.0%); coherence showed no improvement at all (53.3% → 53.3%), suggesting it resists evidence-grounding differently than fact-checkable dimensions.

Combining chain-of-thought with order-averaging showed the debiasing mechanism now works as designed — tie-break outcomes matched the coin flip in only **28% of cases** (near chance, versus 100% under the original prompt) — but 41.7% of comparisons still needed a tie-break, and none were unanimous. **A real, quantified partial fix — not a complete one.** The pointwise judge remains the recommended primary evaluation path; pairwise comparison needs further prompt redesign (e.g. mandating quoted evidence spans) before production use.

### Stress-Testing: Verbosity Bias

Tested whether the pointwise judge rewards verbosity independent of quality, by padding real summaries with hedging filler and a redundant restatement (more words, zero new facts) across 60 comparisons. Result: **0% of padded variants scored higher** than the original; 45% scored strictly lower (mean delta −0.87 on a 1-5 scale). A genuine positive reliability signal, reported as a negative result rather than omitted for not confirming the hypothesis.

---

## Part 2: Scale & Observability Infrastructure

### Pluggable Backend Interface

A `JudgeBackend` protocol decouples the execution layer from any specific model provider. Two implementations: one wrapping real local model calls, and a mock cloud-API simulator (configurable latency distribution, injectable transient-failure rate) that proves out concurrency and retry behavior against realistic failure conditions at zero API cost. Swapping in a real commercial LLM API requires implementing this one interface — no other code changes.

### Async Orchestration: Concurrency, Retries, Checkpointing

An orchestrator runs judge calls concurrently under a bounded worker pool, retries transient failures with linear backoff, and checkpoints automatically — the structured telemetry log itself is the checkpoint, so a killed run resumes by re-invoking with the same job list; completed work is skipped with no separate state file to drift out of sync. Verified with tests asserting *real* concurrency occurred (not just "no exception was raised").

### Structured Observability

Every judge call emits a structured event (latency, parse success, escalation flag, retry count, error) as append-only JSONL — the shape a production observability backend (Prometheus, a data warehouse, Grafana) would ingest. A dashboard generator renders this into a self-contained HTML report: throughput, latency percentiles, per-dimension breakdown, and retry/error detail.

### The Proof Run

Ran 2,000 real jobs (500 real dataset items × 4 dimensions) through the mock cloud backend at 200-way concurrency — roughly 130x the item volume of any single sequential study earlier in the project.

| Metric | Result |
|---|---|
| Throughput | 520-684 jobs/sec |
| Parse success | 100% |
| Retry rate | 2.4-2.9% (matches configured 3% failure rate) |
| Unrecoverable failures | 0% |

**Scaling projection:**

| Workload | Job count | Time |
|---|---|---|
| Full evaluation dataset | 6,400 | 12.3 seconds |
| 10x multi-dataset workload | 64,000 | 2.1 minutes |
| 1M-item production workload | 4,000,000 | ~2.1 hours |

Checkpoint/resume was verified against two independent real process re-invocations: re-running the exact script against an already-completed run finished in **0.01 seconds with zero backend calls**.

### Closing the Loop

The real, calibration-derived abstention policy from Part 1 was wired directly into the orchestrator's escalation logic and re-run at the same 2,000-job scale. Result: **exactly 25.0% escalation** (500/2,000), with fluency escalating 100% of the time and the other three dimensions 0% — precisely reproducing the Part 1 calibration finding, now demonstrated end-to-end through production-shaped infrastructure.

---

## Test Coverage

135 unit tests across 11 modules, all offline and deterministic (no live model server or network access required for CI), including tests that specifically assert real concurrency and real checkpoint behavior rather than just absence-of-error.
