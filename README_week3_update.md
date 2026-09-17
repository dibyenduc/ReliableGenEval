# Week 3: Judge Reliability — Calibration, Position Bias, and Verbosity

This week extended ReliableGenEval from "can we build a judge" to "can we
trust it, and under what conditions" — the core question the project is
named for. All findings below are from real runs against `llama3.1:latest`
on real SummEval articles/summaries, not synthetic data. Code for every
study lives under `src/evalcore/judge/` (pure, unit-tested logic) and
`scripts/` (network-calling study runners); raw outputs are in `results/`.

## 1. Pointwise Judge + Calibration-Driven Abstention

Built a pointwise judge (`evalcore/judge/pointwise.py`) that scores a single
summary on one dimension (coherence, consistency, fluency, relevance) on a
1–5 scale, with a parser (`parse_pointwise_score`) robust to messy LLM output
("2 out of 5", "I'd rate this a 2", bare integers, etc.).

Calibrated against 100 real human-annotated SummEval items
(`scripts/apply_abstention_policy.py`), correlating judge scores against
human scores per dimension:

| Dimension | r | p-value | Verdict |
|---|---|---|---|
| Coherence | 0.550 | 2.98e-09 | Trusted |
| Consistency | 0.551 | 2.93e-09 | Trusted |
| Relevance | 0.483 | 3.69e-07 | Trusted |
| Fluency | 0.155 | 0.1235 | **Escalate to human** |

Built `build_abstention_policy()` to encode this as a reusable rule: escalate
a dimension to human review if its judge-human correlation is not
statistically significant (p ≥ 0.05) OR too weak to be practically useful
(\|r\| < 0.3) even if significant. On this calibration set, that puts 25% of
all judge calls (1 of 4 dimensions) into the escalation bucket — a concrete,
data-driven answer to "when should this system defer to a human?" rather
than a hand-picked threshold.

## 2. Pairwise Judge: Severe, Systematic Position Bias

Built a pairwise judge (`evalcore/judge/pairwise.py`) that compares two
summaries head-to-head on one dimension and returns A/B/Tie, plus a
position-bias test (`evalcore/judge/bias.py`) that runs each comparison in
both presentation orders and checks whether the *preferred summary's
identity* stays consistent regardless of which slot it's shown in.

Run on 15 real SummEval pairs × 4 dimensions (60 comparisons,
`scripts/run_position_bias_study.py`):

| Dimension | Inconsistency rate |
|---|---|
| Coherence | 53.3% |
| Consistency | 86.7% |
| Fluency | 73.3% |
| Relevance | 53.3% |

Overall: **66.7% of comparisons flipped preference purely from swapping
presentation order** — and in every single one of the 40 inconsistent
cases, the judge preferred whichever summary was shown *first*. This isn't
noisy disagreement; it's a deterministic "prefer slot A" heuristic
overriding content on two-thirds of comparisons. Consistency (factual
alignment) was worst, which is notable since it's the dimension most
dependent on actually reading the text carefully.

`summarize_position_bias()` flags any dimension exceeding a 20%
inconsistency threshold as unusable without debiasing
(`scripts/analyze_position_bias.py`). Result: **all 4 dimensions fail this
threshold** — the naive single-order pairwise judge is not currently
trustworthy on any dimension.

## 3. Attempted Fix: Order-Debiasing via Majority Vote — and Why It Fails

The standard mitigation for position bias is to run each comparison in both
orders and take a majority vote (adding a third, differently-ordered call to
break ties). Implemented this in `run_debiased_pairwise_judge()` and
re-ran on the identical 15 pairs.

**First attempt (buggy):** the tie-break call reused the same prompt/order
as the original call. Result: 100% of the 40 tie-break cases resolved to
whichever summary was shown first in the *original* order — the "fix" just
re-injected the same bias with extra steps.

**Second attempt (randomized tie-break order):** fixed to randomize which
order the tie-break call uses. Result was more revealing, not more
reassuring: **44 of 44 tie-break verdicts (100%) exactly matched whichever
order the random coin flip happened to select**, independent of the other
two votes. This means the model's pairwise verdict, in cases where the two
orders disagree, is not a noisy-but-partially-content-driven signal — it is
apparently a **pure function of slot position** with zero recoverable
content signal in the disagreement cases.

**Conclusion:** majority-vote order-debiasing cannot fix this pairwise
judge, because there's no underlying content signal to average toward. The
technique doesn't fail quietly — it functions as a diagnostic that proves
the corruption is total rather than partial for this prompt/model
combination. Fixing this would require redesigning the pairwise prompt
itself (e.g., forcing quoted evidence or step-by-step reasoning before a
verdict) rather than sampling around the existing one. Until then, the
project's default recommendation is: **use the calibration-validated
pointwise judge, not pairwise comparison**, for this model.

## 4. Verbosity/Style Bias: A Negative Result (in a Good Way)

Tested whether the pointwise judge rewards padding/verbosity independent of
content, by constructing a "verbose variant" of real summaries
(`evalcore/judge/style_bias.py`) — adding hedging filler and a verbatim
redundant restatement, with zero new facts — and comparing judge scores on
original vs. padded versions across 15 summaries × 4 dimensions (60
comparisons, `scripts/run_verbosity_bias_study.py`).

| Dimension | Mean score delta (padded − original) | % scored lower |
|---|---|---|
| Relevance | −1.20 | 60.0% |
| Coherence | −0.93 | 46.7% |
| Consistency | −0.80 | 40.0% |
| Fluency | −0.53 | 33.3% |

**0% of padded variants scored higher than the original** — the opposite of
the commonly-cited "LLM judges reward verbosity" failure mode. The judge
consistently penalized filler/redundancy across every dimension, with
relevance penalized most (makes sense — padding is by definition irrelevant)
and fluency least (the added sentences are individually grammatical). This
is a genuine positive reliability signal for the pointwise judge, with the
caveat that it only tests *filler* padding — a follow-up worth doing is
testing whether padding with additional *true, relevant* detail behaves
differently.

## Bottom Line for This Project

| Judge type | Status | Basis |
|---|---|---|
| Pointwise: coherence, consistency, relevance | Trusted | r > 0.48, p < 1e-6 vs. human scores |
| Pointwise: fluency | Escalate to human | r = 0.155, not significant |
| Pointwise: verbosity robustness | Passed | 0/60 padded variants scored higher |
| Pairwise (any dimension) | Not trustworthy | 53–87% position-inconsistency; debiasing fails |

The project's current recommended architecture is: score with the
calibration-validated pointwise judge, escalate fluency judgments to human
review, and avoid pairwise comparison entirely for this model/prompt
combination until the prompt itself is redesigned.

## Test Coverage

87 unit tests across 8 modules (all offline, pure-function tests — no live
Ollama server required for CI): abstention policy, bootstrap CIs,
calibration statistics, pairwise/pointwise parsing, permutation tests,
position-bias resolution and summarization, sensitivity analysis, style-bias
variant generation, and the SummEval data loader.
