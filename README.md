# ReliableGenEval

A human-calibrated framework for statistically sound generative-model comparison
under limited samples, with a production-style observability layer for monitoring
evaluation reliability over time.

**Status**: Weeks 1–2 complete (statistical core + small-sample sensitivity study).
Weeks 3–6 (LLM-as-judge, bias experiments, service layer, monitoring) in progress.

## Key Findings So Far

### 1. Bootstrap CI reliability depends on the metric, not just sample size

We ran a repeated-subsampling sensitivity study (500 trials per sample size, sizes
25/50/100) on the [SummEval](https://github.com/Yale-LILY/SummEval) human-annotated
summarization benchmark, comparing pairs of models across all four human-rated
dimensions (coherence, consistency, fluency, relevance). At each sample size we
measured **empirical coverage**: how often the 95% bootstrap CI actually contained
the true (full-sample) effect.

Replicated across two independent model pairs (model 5 vs. 9, and model 1 vs. 7):

| Metric | n=25 coverage (pair A) | n=25 coverage (pair B) | Nominal | Verdict |
|---|---|---|---|---|
| Coherence | 95.2% | 96.2% | 95% | Well-calibrated |
| Consistency | 82.6% | 84.0% | 95% | **Undercovers** |
| Fluency | 95.2% | 93.6% | 95% | Roughly calibrated |
| Relevance | 95.2% | 97.2% | 95% | Well-calibrated |

**Finding**: the naive percentile bootstrap CI is measurably overconfident on
`consistency` at small sample sizes — actual coverage is 11–15 percentage points
below the nominal 95% level, consistently across two independent model comparisons.
Critically, this is *not* fully explained by "ceiling-clustered" score distributions:
`fluency` is similarly bounded and similarly near-ceiling in this dataset, yet
remains well-calibrated. This means a small-sample eval framework cannot assume a
single calibration correction across metrics — coverage needs to be checked
empirically per metric (and likely per model-pair), not assumed uniform.

### 2. Statistical calibration and statistical power are different properties

At n=25, bootstrap CIs were well-calibrated on real-effect metrics (coherence,
relevance) — but the paired permutation test only detected a real, moderate effect
(Cohen's d ≈ 0.6–0.7) as statistically significant 62–70% of the time. By n=50,
detection jumped to 95–99%. In other words: **a correctly "honest" wide CI at n=25
can still coexist with a one-in-three chance of missing a real model difference
entirely** if you're only looking at a significance threshold. Evaluation pipelines
that report only "model A beats model B, p<0.05" without accounting for this are
systematically underpowered at typical human-eval sample sizes.

## Methodology

- **Dataset**: SummEval (100 human-annotated CNN/DailyMail articles, 16 machine
  summaries each, expert+crowd human ratings on 4 dimensions). Loaded via the free
  `mteb/summeval` Hugging Face mirror, reshaped to long format, validated against
  known dataset invariants before use.
- **Bootstrap CI**: paired percentile bootstrap on mean/median difference, with
  optional cluster-by-article resampling (validated via synthetic data to correctly
  widen CIs when a shared per-article effect diverges between models).
- **Permutation test**: paired sign-flip test (respects pairing; does not assume
  independence between model A's and model B's scores on the same article).
- **Sensitivity study**: for each (metric, model pair, sample size), 500
  independent random subsamples without replacement; each subsample gets its own
  bootstrap CI (2000 resamples) and permutation test (2000 permutations). Empirical
  coverage = fraction of subsample CIs containing the full-sample point estimate.
  Coverage uncertainty reported via binomial standard error.
- **Model pair selection**: pairs were chosen by ranking all 120 possible model
  pairs by Cohen's d on coherence and deliberately selecting from the *median*
  effect-size range (not the largest), so the study measures a realistic, moderate
  effect rather than a trivially detectable one. See `scripts/explore_model_pairs.py`.

## Reproducing the Results

```bash
uv sync
uv run pytest tests/ -v                                   # 25 tests, all offline
uv run python scripts/run_full_sensitivity_study.py --model-a 5 --model-b 9
uv run python scripts/run_full_sensitivity_study.py --model-a 1 --model-b 7
uv run python scripts/plot_sensitivity_results.py
```

Results are written to `results/` as CSVs (raw per-trial and per-metric summaries)
and a comparison plot at `results/sensitivity_plot.png`.

## Project Structure

src/evalcore/
data/summeval.py # dataset ingestion, reshape, validation
stats/bootstrap.py # paired bootstrap CI (naive + cluster-by-article)
stats/permutation.py # paired sign-flip permutation test
analysis/sensitivity.py # repeated-subsampling small-sample sensitivity study
scripts/
explore_model_pairs.py # rank model pairs by effect size
run_full_sensitivity_study.py # run sensitivity study for a given model pair
plot_sensitivity_results.py # visualize CI width / coverage vs sample size
tests/ # pytest unit tests for all statistical functions
results/ # generated CSVs and plots (not hand-edited)
.github/workflows/ci.yml # pytest + ruff lint on every push/PR


## Limitations and Threats to Validity (so far)

- The sensitivity study uses only 2 of 120 possible SummEval model pairs; the
  consistency-undercoverage finding is replicated twice but not exhaustively
  verified across all pairs or all effect-size regimes.
- SummEval's full per-model sample size is capped at 100 (one summary per article),
  since human annotation is expensive — this is a realistic constraint, but it
  means "n=100" in our study is a ceiling, not an arbitrarily large reference.
- We have not yet identified *why* consistency specifically undercovers while
  fluency does not, despite both being bounded/skewed metrics — this is flagged as
  an open question for the technical report rather than resolved here.
- Naive (non-cluster) bootstrap was used for the sensitivity study; cluster-by-
  article resampling exists in `evalcore.stats.bootstrap` but was not the default
  used in this particular experiment — a documented, deliberate scope choice for
  Week 2, revisited if it changes conclusions in later analysis.

## Roadmap

- **Week 3**: LLM-as-judge wrapper (local Ollama model), judge-vs-human calibration,
  bias stress tests (position, verbosity, style), human-escalation/abstention rule.
- **Week 4**: FastAPI service replaying SummEval as a live event stream, scoring
  in real time, logging to SQLite.
- **Week 5**: Background drift monitor (KS test / PSI) with alerting tied to
  offline-derived thresholds, Streamlit dashboard.
- **Week 6**: CI drift-injection test, technical report/decision memo, final
  packaging.

## License

MIT (see LICENSE)
